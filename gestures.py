# --- keep logs quiet (must be set BEFORE importing mediapipe/tflite) ---
import os
os.environ["GLOG_minloglevel"] = "2"   # 0=INFO,1=WARNING,2=ERROR,3=FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time, collections, numpy as np, cv2
from picamera2 import Picamera2
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)

# MediaPipe for pinch
import mediapipe as mp
mp_hands = mp.solutions.hands

# =========================
# Ultra-fast LR/UD swipes via column/row motion energy + pinch
# =========================
FRAME_W, FRAME_H = 640, 480
LO_W, LO_H       = 192, 108

# Horizontal band (rows in lores, normalized)
ROI_Y0, ROI_Y1 = 0.25, 0.75
# Vertical band (cols in lores, normalized)
ROI_X0, ROI_X1 = 0.25, 0.75

# Horizontal gates
CROSS_L, CROSS_R = 0.12, 0.88
HYST    = 0.02
L_ARM, L_FIRE = CROSS_L - HYST, CROSS_L + HYST
R_ARM, R_FIRE = CROSS_R + HYST, CROSS_R - HYST

# Vertical gates
CROSS_T, CROSS_B = 0.12, 0.88
T_ARM, T_FIRE = CROSS_T - HYST, CROSS_T + HYST
B_ARM, B_FIRE = CROSS_B + HYST, CROSS_B - HYST

COOLDOWN_S, ABSENCE_RESET_S = 0.12, 0.10

# Span/velocity windows (normalized widths/heights per second)
SPAN_WINDOW_S = 0.28
SPAN_THR_X, VEL_THR_X = 0.45, 2.2
SPAN_THR_Y, VEL_THR_Y = 0.45, 2.2

ENERGY_MIN_FRAC = 0.015
SMOOTH_COLS, SMOOTH_ROWS = 5, 5
POOL_FRAMES = 2

# Pinch
PINCH_RATIO_LOW, PINCH_RATIO_HIGH = 0.45, 0.55
PINCH_EVERY_N_FRAMES = 2          # run MP every N frames
PINCH_SQUARE_SIZE    = 256        # send square to MP to avoid ROI warnings
IGNORE_SWIPES_WHEN_PINCH = True
SHOW_PINCH_RATIO = True

HEADLESS = os.environ.get("DISPLAY", "") == ""

# =========================
# Helpers
# =========================
def y_plane(yuv):
    # YUV420 planar: top LO_H rows are luma
    if yuv.ndim == 2:  return yuv[:LO_H, :LO_W]
    if yuv.ndim == 3:  return yuv[:, :, 0]
    raise ValueError(f"Unexpected lores shape {yuv.shape}")

def smooth1d(v, k):
    if k <= 0: return v
    ksz = 2*k + 1
    ker = np.ones(ksz, np.float32) / ksz
    return np.convolve(v, ker, mode="same")

def pinch_ratio(lm):
    # thumb tip (4), index tip (8); scale by wrist(0)->middle MCP(9)
    t, i = lm[4], lm[8]; w, m = lm[0], lm[9]
    d_tip  = np.hypot(t.x - i.x, t.y - i.y)
    d_size = np.hypot(w.x - m.x, w.y - m.y) + 1e-6
    return d_tip / d_size

def center_square_crop_rgb(rgb, out_size):
    h, w, _ = rgb.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    sq = rgb[y0:y0+side, x0:x0+side]
    if side != out_size:
        sq = cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return sq

# =========================
# Camera
# =========================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format":"RGB888", "size": (FRAME_W, FRAME_H)},
    lores={"format":"YUV420", "size": (LO_W, LO_H)},
    display="main",
)
picam2.configure(config)
try:
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True, "AfMode": 2,
        "FrameDurationLimits": (10000, 10000),  # ~100 fps (device may cap lower)
        "AnalogueGain": 12.0,
    })
except Exception:
    pass
picam2.start()

# =========================
# State
# =========================
prev_lo = None
mask_pool = collections.deque(maxlen=POOL_FRAMES)
trace_x, trace_y = collections.deque(), collections.deque()
state_x, state_y = "IDLE", "IDLE"
last_fire = 0.0
last_seen_t_x = last_seen_t_y = 0.0

pinch_state = False
pinch_ratio_val = None
frame_idx = 0

sx, sy = FRAME_W/LO_W, FRAME_H/LO_H
y0, y1 = int(ROI_Y0*LO_H), int(ROI_Y1*LO_H)
x0, x1 = int(ROI_X0*LO_W), int(ROI_X1*LO_W)

# MediaPipe Hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60,
)

try:
    while True:
        now = time.time(); frame_idx += 1

        # Frames
        lo  = y_plane(picam2.capture_array("lores"))
        rgb = picam2.capture_array("main")        # RGB for MP + preview
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        x_norm = y_norm = None

        # ---- Pinch (square crop; warnings silenced above) ----
        if frame_idx % PINCH_EVERY_N_FRAMES == 0:
            rgb_sq = center_square_crop_rgb(rgb, PINCH_SQUARE_SIZE)
            res = hands.process(rgb_sq)  # expects RGB array
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                pr = pinch_ratio(lm)
                pinch_ratio_val = pr
                if not pinch_state and pr < PINCH_RATIO_LOW:
                    pinch_state = True; print("PINCH_START")
                elif pinch_state and pr > PINCH_RATIO_HIGH:
                    pinch_state = False; print("PINCH_END")

        # ---- Motion energy paths ----
        if prev_lo is not None:
            diff = cv2.absdiff(lo, prev_lo)
            mask_pool.append(diff)
            pooled = mask_pool[0]
            for i in range(1, len(mask_pool)):
                pooled = cv2.bitwise_or(pooled, mask_pool[i])

            # Horizontal (columns in vertical band)
            band_h = pooled[y0:y1, :]
            col = band_h.astype(np.float32).sum(axis=0)
            if col.sum() >= ENERGY_MIN_FRAC * (255.0 * (y1-y0) * LO_W):
                col_s = smooth1d(col, SMOOTH_COLS//2)
                xs = np.arange(LO_W, dtype=np.float32)
                wsum = col_s.sum()
                if wsum > 1e-3:
                    cx = float((col_s * xs).sum() / wsum)
                    x_norm = cx / LO_W
                    # inset
                    bar = (col_s / (col_s.max()+1e-6) * 255.0).astype(np.uint8)
                    bar = np.tile(bar, (40,1))
                    bar = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
                    bar = cv2.resize(bar, (FRAME_W, 40))
                    dbg[0:40, 0:FRAME_W] = bar
                    cv2.line(dbg, (int(x_norm*FRAME_W), 40), (int(x_norm*FRAME_W), 70), (255,255,255), 2)

            # Vertical (rows in horizontal band)
            band_v = pooled[:, x0:x1]
            row = band_v.astype(np.float32).sum(axis=1)
            if row.sum() >= ENERGY_MIN_FRAC * (255.0 * (x1-x0) * LO_H):
                row_s = smooth1d(row, SMOOTH_ROWS//2)
                ys = np.arange(LO_H, dtype=np.float32)
                wsum = row_s.sum()
                if wsum > 1e-3:
                    cy = float((row_s * ys).sum() / wsum)
                    y_norm = cy / LO_H
                    # inset
                    barv = (row_s / (row_s.max()+1e-6) * 255.0).astype(np.uint8)
                    barv = np.tile(barv[:, None], (1, 40))
                    barv = cv2.cvtColor(barv, cv2.COLOR_GRAY2BGR)
                    barv = cv2.resize(barv, (40, FRAME_H))
                    dbg[0:FRAME_H, FRAME_W-40:FRAME_W] = barv
                    cv2.line(dbg, (FRAME_W-40, int(y_norm*FRAME_H)), (FRAME_W, int(y_norm*FRAME_H)), (255,255,255), 2)

        prev_lo = lo

        # Trace windows
        while trace_x and (now - trace_x[0][0]) > SPAN_WINDOW_S: trace_x.popleft()
        while trace_y and (now - trace_y[0][0]) > SPAN_WINDOW_S: trace_y.popleft()
        if x_norm is not None: trace_x.append((now, x_norm)); last_seen_t_x = now
        if y_norm is not None: trace_y.append((now, y_norm)); last_seen_t_y = now

        # Spans / velocities
        span_x = vel_x = 0.0
        if len(trace_x) >= 2:
            xs = [p[1] for p in trace_x]; ts = [p[0] for p in trace_x]
            span_x = max(xs) - min(xs); dt = max(1e-3, ts[-1]-ts[0]); vel_x = (xs[-1]-xs[0])/dt
        span_y = vel_y = 0.0
        if len(trace_y) >= 2:
            ys = [p[1] for p in trace_y]; ts2 = [p[0] for p in trace_y]
            span_y = max(ys) - min(ys); dt2 = max(1e-3, ts2[-1]-ts2[0]); vel_y = (ys[-1]-ys[0])/dt2

        # Decide swipes
        gesture_text = ""
        can_fire = (now - last_fire) > COOLDOWN_S
        pinch_block = (IGNORE_SWIPES_WHEN_PINCH and pinch_state)

        # Horizontal gates
        if not pinch_block and x_norm is not None:
            if state_x == "IDLE":
                if x_norm <= L_ARM: state_x = "ARM_RIGHT"
                elif x_norm >= R_ARM: state_x = "ARM_LEFT"
            if can_fire:
                if state_x == "ARM_RIGHT" and x_norm >= R_FIRE:
                    print("SWIPE_RIGHT"); gesture_text = "SWIPE_RIGHT"; last_fire = now; state_x = "IDLE"; trace_x.clear()
                elif state_x == "ARM_LEFT" and x_norm <= L_FIRE:
                    print("SWIPE_LEFT");  gesture_text = "SWIPE_LEFT";  last_fire = now; state_x = "IDLE"; trace_x.clear()
        else:
            if state_x != "IDLE" and (now - last_seen_t_x) > ABSENCE_RESET_S: state_x = "IDLE"

        # Vertical gates
        if not pinch_block and y_norm is not None and gesture_text == "":
            if state_y == "IDLE":
                if y_norm <= T_ARM: state_y = "ARM_DOWN"
                elif y_norm >= B_ARM: state_y = "ARM_UP"
            if can_fire:
                if state_y == "ARM_DOWN" and y_norm >= B_FIRE:
                    print("SWIPE_DOWN"); gesture_text = "SWIPE_DOWN"; last_fire = now; state_y = "IDLE"; trace_y.clear()
                elif state_y == "ARM_UP" and y_norm <= T_FIRE:
                    print("SWIPE_UP");   gesture_text = "SWIPE_UP";   last_fire = now; state_y = "IDLE"; trace_y.clear()
        else:
            if state_y != "IDLE" and (now - last_seen_t_y) > ABSENCE_RESET_S: state_y = "IDLE"

        # Span+velocity fallback
        if (not pinch_block) and can_fire and gesture_text == "" and span_x >= SPAN_THR_X and abs(vel_x) >= VEL_THR_X:
            print("SWIPE_RIGHT" if vel_x>0 else "SWIPE_LEFT")
            gesture_text = "SWIPE_RIGHT" if vel_x>0 else "SWIPE_LEFT"
            last_fire = now; state_x = "IDLE"; trace_x.clear()
        if (not pinch_block) and can_fire and gesture_text == "" and span_y >= SPAN_THR_Y and abs(vel_y) >= VEL_THR_Y:
            print("SWIPE_DOWN" if vel_y>0 else "SWIPE_UP")
            gesture_text = "SWIPE_DOWN" if vel_y>0 else "SWIPE_UP"
            last_fire = now; state_y = "IDLE"; trace_y.clear()

        # HUD
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.rectangle(dbg, (0, int(ROI_Y0*FRAME_H)), (FRAME_W, int(ROI_Y1*FRAME_H)), (255,255,255), 1)

        cv2.line(dbg, (0, int(CROSS_T*FRAME_H)), (FRAME_W, int(CROSS_T*FRAME_H)), (255,255,255), 1)
        cv2.line(dbg, (0, int(CROSS_B*FRAME_H)), (FRAME_W, int(CROSS_B*FRAME_H)), (255,255,255), 1)
        cv2.rectangle(dbg, (int(ROI_X0*FRAME_W), 0), (int(ROI_X1*FRAME_W), FRAME_H), (255,255,255), 1)

        x_txt = f"x={trace_x[-1][1]:.2f}" if trace_x else "x=--"
        y_txt = f"y={trace_y[-1][1]:.2f}" if trace_y else "y=--"
        pinch_txt = f"PINCH:{'YES' if pinch_state else 'NO'}"
        if SHOW_PINCH_RATIO and pinch_ratio_val is not None:
            pinch_txt += f" ({pinch_ratio_val:.2f})"

        cv2.putText(dbg, f"{x_txt}  STATE_X={state_x}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(dbg, f"{y_txt}  STATE_Y={state_y}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(dbg, pinch_txt, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Swipes (LR+UD) + Pinch", dbg)
            if cv2.waitKey(1) & 0xFF == 27: break
        else:
            time.sleep(0.002)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
