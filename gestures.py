import time, os, collections
import numpy as np
import cv2
from picamera2 import Picamera2

# =========================
# Ultra-fast swipe via column motion energy
# =========================
FRAME_W, FRAME_H = 640, 480     # preview
LO_W, LO_H       = 192, 108     # tiny lores (high FPS & low latency)

# Detection band (normalized rows of lores frame)
ROI_Y0 = 0.25
ROI_Y1 = 0.75

# Gates & timing
CROSS_L = 0.12
CROSS_R = 0.88
HYST    = 0.02
L_ARM, L_FIRE = CROSS_L - HYST, CROSS_L + HYST
R_ARM, R_FIRE = CROSS_R + HYST, CROSS_R - HYST

COOLDOWN_S       = 0.12         # quick repeats
ABSENCE_RESET_S  = 0.10
SPAN_WINDOW_S    = 0.20         # window to consider span
SPAN_THR         = 0.55         # must cover ≥ 55% width in window
VEL_THR          = 3.5          # normalized widths/sec (|dx|/dt) threshold
ENERGY_MIN_FRAC  = 0.015        # min total motion energy to accept frame

# 1D smoothing over columns (frames) to stabilize peak
SMOOTH_COLS      = 5            # box filter half-width (odd -> kernel=2k+1)
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

def col_smooth(v, k):
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
mask_pool = collections.deque(maxlen=POOL_FRAMES)
trace = collections.deque()  # (t, x_norm)
state = "IDLE"
last_fire = 0.0
last_seen_t = 0.0

sx, sy = FRAME_W / LO_W, FRAME_H / LO_H
y0 = int(ROI_Y0 * LO_H)
y1 = int(ROI_Y1 * LO_H)

try:
    while True:
        now = time.time()

        # ---- Get frames ----
        lo = y_plane(picam2.capture_array("lores"))
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        x_norm = None

        if prev_lo is not None:
            # ---- Raw motion ----
            diff = cv2.absdiff(lo, prev_lo)  # uint8

            # Restrict to ROI band
            band = diff[y0:y1, :]

            # Temporal pooling (OR of last few frames) to keep thin/fast signals
            mask_pool.append(band)
            pooled = mask_pool[0]
            for i in range(1, len(mask_pool)):
                pooled = cv2.bitwise_or(pooled, mask_pool[i])

            # Column motion energy
            col = pooled.astype(np.float32).sum(axis=0)  # shape: (LO_W,)
            total_energy = col.sum()

            if total_energy >= ENERGY_MIN_FRAC * (255.0 * (y1 - y0) * LO_W):
                # Smooth column energy, then get weighted centroid (peak center)
                col_s = col_smooth(col, SMOOTH_COLS//2)
                xs = np.arange(LO_W, dtype=np.float32)
                wsum = col_s.sum()
                if wsum > 1e-3:
                    cx = float((col_s * xs).sum() / wsum)
                    x_norm = cx / LO_W

                    # draw column heat as tiny inset for debugging
                    bar = (col_s / (col_s.max() + 1e-6) * 255.0).astype(np.uint8)
                    bar = np.tile(bar, (40,1))
                    bar = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
                    bar = cv2.resize(bar, (FRAME_W, 60))
                    dbg[0:60, 0:FRAME_W] = bar

                    # show centroid
                    cv2.line(dbg, (int(x_norm*FRAME_W), 60), (int(x_norm*FRAME_W), 60+40), (255,255,255), 2)

        prev_lo = lo

        # ---- Maintain short window of positions ----
        while trace and (now - trace[0][0]) > SPAN_WINDOW_S:
            trace.popleft()
        if x_norm is not None:
            trace.append((now, x_norm))
            last_seen_t = now

        # ---- Decide swipe: gated OR span+velocity ----
        gesture_text = ""
        can_fire = (now - last_fire) > COOLDOWN_S

        # State machine arming/firing using current x
        if x_norm is not None:
            if state == "IDLE":
                if x_norm <= L_ARM:
                    state = "ARM_RIGHT"
                elif x_norm >= R_ARM:
                    state = "ARM_LEFT"

            if can_fire:
                if state == "ARM_RIGHT" and x_norm >= R_FIRE:
                    print("SWIPE_RIGHT")
                    gesture_text = "SWIPE_RIGHT"
                    last_fire = now
                    state = "IDLE"
                elif state == "ARM_LEFT" and x_norm <= L_FIRE:
                    print("SWIPE_LEFT")
                    gesture_text = "SWIPE_LEFT"
                    last_fire = now
                    state = "IDLE"
        else:
            if state != "IDLE" and (now - last_seen_t) > ABSENCE_RESET_S:
                state = "IDLE"

        # Span + velocity (handles mid-screen appearances + single-frame misses)
        if can_fire and gesture_text == "" and len(trace) >= 2:
            xs = [p[1] for p in trace]
            ts = [p[0] for p in trace]
            span = max(xs) - min(xs)
            dt = max(1e-3, ts[-1] - ts[0])
            v = (xs[-1] - xs[0]) / dt  # widths/sec (normalized)

            if span >= SPAN_THR and abs(v) >= VEL_THR:
                if v > 0:
                    print("SWIPE_RIGHT")
                    gesture_text = "SWIPE_RIGHT"
                else:
                    print("SWIPE_LEFT")
                    gesture_text = "SWIPE_LEFT"
                last_fire = now
                state = "IDLE"
                trace.clear()   # prevent double fire

        # ---- Draw HUD ----
        # Gates
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)
        # ROI band (on preview scale)
        cv2.rectangle(dbg, (0, int(ROI_Y0*FRAME_H)), (FRAME_W, int(ROI_Y1*FRAME_H)), (255,255,255), 1)

        # Status text
        x_txt = f"x={trace[-1][1]:.2f}" if trace else "x=--"
        cv2.putText(dbg, f"{x_txt}  STATE={state}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Swipe (ultrafast column energy)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.002)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
