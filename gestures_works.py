import time, os, collections
import numpy as np
import cv2
from picamera2 import Picamera2

# =========================
# Ultra-fast swipe via column/row motion energy (LR + UD)
# =========================
FRAME_W, FRAME_H = 640, 480     # preview
LO_W, LO_H       = 192, 108     # tiny lores (high FPS & low latency)

# Horizontal detection band (rows in lores, normalized 0..1)
ROI_Y0 = 0.25
ROI_Y1 = 0.75

# Vertical detection band (cols in lores, normalized 0..1)
ROI_X0 = 0.25
ROI_X1 = 0.75

# Gates & timing (horizontal)
CROSS_L = 0.12
CROSS_R = 0.88
HYST    = 0.02
L_ARM, L_FIRE = CROSS_L - HYST, CROSS_L + HYST
R_ARM, R_FIRE = CROSS_R + HYST, CROSS_R - HYST

# Gates & timing (vertical)
CROSS_T = 0.12
CROSS_B = 0.88
T_ARM, T_FIRE = CROSS_T - HYST, CROSS_T + HYST
B_ARM, B_FIRE = CROSS_B + HYST, CROSS_B - HYST

COOLDOWN_S       = 0.12         # quick repeats (shared across axes)
ABSENCE_RESET_S  = 0.10

# Span/velocity windows (normalized units per second)
SPAN_WINDOW_S    = 0.28
SPAN_THR_X       = 0.45
VEL_THR_X        = 2.2
SPAN_THR_Y       = 0.45
VEL_THR_Y        = 2.2

ENERGY_MIN_FRAC  = 0.015        # min total motion energy to accept frame

# 1D smoothing over energy curves
SMOOTH_COLS      = 5            # for horizontal (columns)
SMOOTH_ROWS      = 5            # for vertical (rows)
POOL_FRAMES      = 2            # OR of last N diff maps (tiny temporal pool)

HEADLESS = os.environ.get("DISPLAY", "") == ""

# =========================
# Helpers
# =========================
def y_plane(yuv):
    # Picamera2 YUV420 planar: take top LO_H rows
    if yuv.ndim == 2:
        return yuv[:LO_H, :LO_W]
    elif yuv.ndim == 3:
        return yuv[:, :, 0]
    raise ValueError(f"Unexpected lores shape {yuv.shape}")

def smooth1d(v, k):
    if k <= 0: return v
    ksz = 2*k + 1
    kernel = np.ones(ksz, dtype=np.float32) / ksz
    return np.convolve(v, kernel, mode="same")

# =========================
# Camera
# =========================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (FRAME_W, FRAME_H)},
    lores={"format": "YUV420", "size": (LO_W, LO_H)},
    display="main",
)
picam2.configure(config)

# Push FPS aggressively; compensate with gain (tune if too dark)
try:
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True,
        "AfMode": 2,
        "FrameDurationLimits": (10000, 10000),   # ~100 fps (falls back if not supported)
        "AnalogueGain": 12.0,
    })
except Exception:
    pass

picam2.start()

# =========================
# State
# =========================
prev_lo = None
mask_pool = collections.deque(maxlen=POOL_FRAMES)   # pooled diff (for both bands)
trace_x = collections.deque()  # (t, x_norm) for LR
trace_y = collections.deque()  # (t, y_norm) for UD

state_x = "IDLE"               # IDLE | ARM_RIGHT | ARM_LEFT
state_y = "IDLE"               # IDLE | ARM_DOWN | ARM_UP
last_fire = 0.0
last_seen_t_x = 0.0
last_seen_t_y = 0.0

sx, sy = FRAME_W / LO_W, FRAME_H / LO_H
y0 = int(ROI_Y0 * LO_H)
y1 = int(ROI_Y1 * LO_H)
x0 = int(ROI_X0 * LO_W)
x1 = int(ROI_X1 * LO_W)

try:
    while True:
        now = time.time()

        # ---- Get frames ----
        lo = y_plane(picam2.capture_array("lores"))
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        x_norm = None
        y_norm = None

        if prev_lo is not None:
            # ---- Raw motion ----
            diff = cv2.absdiff(lo, prev_lo)  # uint8

            # Temporal pooling (OR of last few frames) to keep thin/fast signals
            mask_pool.append(diff)
            pooled = mask_pool[0]
            for i in range(1, len(mask_pool)):
                pooled = cv2.bitwise_or(pooled, mask_pool[i])

            # --- Horizontal path: column motion energy in vertical band (rows y0:y1)
            band_h = pooled[y0:y1, :]
            col = band_h.astype(np.float32).sum(axis=0)  # (LO_W,)
            total_energy_h = col.sum()

            if total_energy_h >= ENERGY_MIN_FRAC * (255.0 * (y1 - y0) * LO_W):
                col_s = smooth1d(col, SMOOTH_COLS//2)
                xs = np.arange(LO_W, dtype=np.float32)
                wsum = col_s.sum()
                if wsum > 1e-3:
                    cx = float((col_s * xs).sum() / wsum)
                    x_norm = cx / LO_W
                    # draw column heat inset
                    bar = (col_s / (col_s.max() + 1e-6) * 255.0).astype(np.uint8)
                    bar = np.tile(bar, (40,1))
                    bar = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
                    bar = cv2.resize(bar, (FRAME_W, 40))
                    dbg[0:40, 0:FRAME_W] = bar
                    cv2.line(dbg, (int(x_norm*FRAME_W), 40), (int(x_norm*FRAME_W), 40+30), (255,255,255), 2)

            # --- Vertical path: row motion energy in horizontal band (cols x0:x1)
            band_v = pooled[:, x0:x1]
            row = band_v.astype(np.float32).sum(axis=1)  # (LO_H,)
            total_energy_v = row.sum()

            if total_energy_v >= ENERGY_MIN_FRAC * (255.0 * (x1 - x0) * LO_H):
                row_s = smooth1d(row, SMOOTH_ROWS//2)
                ys = np.arange(LO_H, dtype=np.float32)
                wsum = row_s.sum()
                if wsum > 1e-3:
                    cy = float((row_s * ys).sum() / wsum)
                    y_norm = cy / LO_H
                    # draw row heat inset (right edge)
                    barv = (row_s / (row_s.max() + 1e-6) * 255.0).astype(np.uint8)
                    barv = np.tile(barv[:, None], (1, 40))
                    barv = cv2.cvtColor(barv, cv2.COLOR_GRAY2BGR)
                    barv = cv2.resize(barv, (40, FRAME_H))
                    dbg[0:FRAME_H, FRAME_W-40:FRAME_W] = barv
                    cv2.line(dbg, (FRAME_W-40, int(y_norm*FRAME_H)), (FRAME_W, int(y_norm*FRAME_H)), (255,255,255), 2)

        prev_lo = lo

        # ---- Maintain short windows of positions ----
        while trace_x and (now - trace_x[0][0]) > SPAN_WINDOW_S:
            trace_x.popleft()
        while trace_y and (now - trace_y[0][0]) > SPAN_WINDOW_S:
            trace_y.popleft()

        if x_norm is not None:
            trace_x.append((now, x_norm))
            last_seen_t_x = now
        if y_norm is not None:
            trace_y.append((now, y_norm))
            last_seen_t_y = now

        # Compute spans & velocities
        span_x = 0.0; vel_x = 0.0
        if len(trace_x) >= 2:
            xs = [p[1] for p in trace_x]; ts = [p[0] for p in trace_x]
            span_x = max(xs) - min(xs)
            dt = max(1e-3, ts[-1] - ts[0]); vel_x = (xs[-1] - xs[0]) / dt

        span_y = 0.0; vel_y = 0.0
        if len(trace_y) >= 2:
            ys = [p[1] for p in trace_y]; ts2 = [p[0] for p in trace_y]
            span_y = max(ys) - min(ys)
            dt2 = max(1e-3, ts2[-1] - ts2[0]); vel_y = (ys[-1] - ys[0]) / dt2

        # ---- Decide swipes (share cooldown across axes) ----
        gesture_text = ""
        can_fire = (now - last_fire) > COOLDOWN_S

        # Horizontal gates
        if x_norm is not None:
            if state_x == "IDLE":
                if x_norm <= L_ARM: state_x = "ARM_RIGHT"
                elif x_norm >= R_ARM: state_x = "ARM_LEFT"
            if can_fire:
                if state_x == "ARM_RIGHT" and x_norm >= R_FIRE:
                    print("SWIPE_RIGHT"); gesture_text = "SWIPE_RIGHT"; last_fire = now; state_x = "IDLE"; trace_x.clear()
                elif state_x == "ARM_LEFT" and x_norm <= L_FIRE:
                    print("SWIPE_LEFT");  gesture_text = "SWIPE_LEFT";  last_fire = now; state_x = "IDLE"; trace_x.clear()
        else:
            if state_x != "IDLE" and (now - last_seen_t_x) > ABSENCE_RESET_S:
                state_x = "IDLE"

        # Vertical gates
        if y_norm is not None and gesture_text == "":
            if state_y == "IDLE":
                if y_norm <= T_ARM: state_y = "ARM_DOWN"
                elif y_norm >= B_ARM: state_y = "ARM_UP"
            if can_fire:
                if state_y == "ARM_DOWN" and y_norm >= B_FIRE:
                    print("SWIPE_DOWN"); gesture_text = "SWIPE_DOWN"; last_fire = now; state_y = "IDLE"; trace_y.clear()
                elif state_y == "ARM_UP" and y_norm <= T_FIRE:
                    print("SWIPE_UP");   gesture_text = "SWIPE_UP";   last_fire = now; state_y = "IDLE"; trace_y.clear()
        else:
            if state_y != "IDLE" and (now - last_seen_t_y) > ABSENCE_RESET_S:
                state_y = "IDLE"

        # Span + velocity fallbacks (mid-screen lightning flicks)
        if can_fire and gesture_text == "" and span_x >= SPAN_THR_X and abs(vel_x) >= VEL_THR_X:
            if vel_x > 0: print("SWIPE_RIGHT"); gesture_text = "SWIPE_RIGHT"
            else:         print("SWIPE_LEFT");  gesture_text = "SWIPE_LEFT"
            last_fire = now; state_x = "IDLE"; trace_x.clear()

        if can_fire and gesture_text == "" and span_y >= SPAN_THR_Y and abs(vel_y) >= VEL_THR_Y:
            if vel_y > 0: print("SWIPE_DOWN"); gesture_text = "SWIPE_DOWN"
            else:         print("SWIPE_UP");   gesture_text = "SWIPE_UP"
            last_fire = now; state_y = "IDLE"; trace_y.clear()

        # ---- HUD ----
        # Horizontal gates + band
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.rectangle(dbg, (0, int(ROI_Y0*FRAME_H)), (FRAME_W, int(ROI_Y1*FRAME_H)), (255,255,255), 1)
        # Vertical gates + band
        cv2.line(dbg, (0, int(CROSS_T*FRAME_H)), (FRAME_W, int(CROSS_T*FRAME_H)), (255,255,255), 1)
        cv2.line(dbg, (0, int(CROSS_B*FRAME_H)), (FRAME_W, int(CROSS_B*FRAME_H)), (255,255,255), 1)
        cv2.rectangle(dbg, (int(ROI_X0*FRAME_W), 0), (int(ROI_X1*FRAME_W), FRAME_H), (255,255,255), 1)

        # Status
        x_txt = f"x={trace_x[-1][1]:.2f}" if trace_x else "x=--"
        y_txt = f"y={trace_y[-1][1]:.2f}" if trace_y else "y=--"
        cv2.putText(dbg, f"{x_txt}  STATE_X={state_x}  spanX={span_x:.2f} velX={vel_x:.2f}",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(dbg, f"{y_txt}  STATE_Y={state_y}  spanY={span_y:.2f} velY={vel_y:.2f}",
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Swipe (ultrafast LR+UD)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.002)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
