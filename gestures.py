import time, os, collections
import numpy as np
import cv2
from picamera2 import Picamera2

# =========================
# Ultra-fast hand swipe
# =========================
FRAME_W, FRAME_H = 640, 480       # preview
LO_W, LO_H       = 224, 128       # smaller = faster (good enough for x-tracking)

# Gates & timing
CROSS_L = 0.12
CROSS_R = 0.88
HYST    = 0.015
L_ARM, L_FIRE = CROSS_L - HYST, CROSS_L + HYST
R_ARM, R_FIRE = CROSS_R + HYST, CROSS_R - HYST

COOLDOWN_S       = 0.14           # allow very quick successive swipes
ABSENCE_RESET_S  = 0.15           # reset quickly if blob disappears
SPAN_WINDOW_S    = 0.22           # max time window to consider span
SPAN_THR         = 0.55           # must cover >=55% of frame width (normalized)
VEL_THR          = 3.0            # normalized widths/sec (|dx|/dt) to count as a flick

# Blob acceptance (loose hand)
MIN_H_FRAC     = 0.45
MIN_W_FRAC     = 0.06
MAX_W_FRAC     = 0.95
MIN_AREA_FRAC  = 0.04
MAX_AREA_FRAC  = 0.92
AR_MIN         = 0.75             # allow nearly square with motion blur

# Motion / morphology
MOG_HISTORY = 40
MOG_VAR_THR = 16
WARMUP_FRAMES = 12
ADAPT_K   = 1.4
THR_FLOOR = 10
KERNEL    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
POOL_FRAMES = 3                    # temporal max-pool of N masks

HEADLESS = os.environ.get("DISPLAY", "") == ""

# =========================
# Helpers
# =========================
def luma_from_lores(yuv_lo):
    if yuv_lo.ndim == 2:
        return yuv_lo[:LO_H, :LO_W]
    elif yuv_lo.ndim == 3:
        return yuv_lo[:, :, 0]
    raise ValueError(f"Unexpected lores shape: {yuv_lo.shape}")

# =========================
# Camera: dual-stream
# =========================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format":"RGB888", "size": (FRAME_W, FRAME_H)},
    lores={"format":"YUV420", "size": (LO_W, LO_H)},
    display="main",
)
picam2.configure(config)

try:
    # Sports-y: short exposure, higher gain, high FPS
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True, "AfMode": 2,
        "FrameDurationLimits": (16666, 16666),   # ~60 fps
        "AnalogueGain": 10.0,
    })
except Exception:
    pass

picam2.start()

# =========================
# State
# =========================
prev_lo = None
mog = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY, varThreshold=MOG_VAR_THR, detectShadows=False)
last_fire = 0.0
state = "IDLE"    # IDLE | ARM_RIGHT | ARM_LEFT
last_seen_t = 0.0

sx, sy = FRAME_W / LO_W, FRAME_H / LO_H

# Temporal pooling of masks (keep last N)
mask_pool = collections.deque(maxlen=POOL_FRAMES)

# Recent positions for span/velocity (deque of (t, x_norm))
trace = collections.deque()

try:
    while True:
        now = time.time()

        # ---- Fast detection (lores Y) ----
        yuv_lo = picam2.capture_array("lores")
        lo_y   = luma_from_lores(yuv_lo)

        # ---- Preview (main) ----
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        cx_norm = None
        bbox_main = None

        if prev_lo is not None:
            diff = cv2.absdiff(lo_y, prev_lo)

            # Adaptive threshold
            m, s = cv2.meanStdDev(diff)
            thr = max(THR_FLOOR, float(m[0][0] + ADAPT_K * s[0][0]))
            _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

            # Morphology
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, KERNEL, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, KERNEL, iterations=2)

            # Background subtractor (low LR after warmup)
            lr = 0.5 if len(mask_pool) < WARMUP_FRAMES else 0.01
            fg = mog.apply(lo_y, learningRate=lr)

            # Combine: AND motion from diff with foreground model; then temporal max-pool
            motion = cv2.bitwise_and(th, fg)
            mask_pool.append(motion)
            pooled = mask_pool[0].copy()
            for i in range(1, len(mask_pool)):
                pooled = cv2.bitwise_or(pooled, mask_pool[i])

            # A little dilation to merge fragmented hand
            pooled = cv2.dilate(pooled, KERNEL, iterations=1)

            # Find candidate blob (largest acceptable)
            cnts, _ = cv2.findContours(pooled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lo_area = LO_W * LO_H
            min_area_px = MIN_AREA_FRAC * lo_area
            max_area_px = MAX_AREA_FRAC * lo_area

            candidate = None
            best_area = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if not (min_area_px <= area <= max_area_px):
                    continue
                h_frac = h / LO_H
                w_frac = w / LO_W
                if h_frac < MIN_H_FRAC:
                    continue
                if not (MIN_W_FRAC <= w_frac <= MAX_W_FRAC):
                    continue
                ar = h / max(1, w)
                if ar < AR_MIN:
                    continue

                if area > best_area:
                    best_area = area
                    candidate = (x, y, w, h)

            if candidate:
                x, y, w, h = candidate
                cx_norm = (x + w/2) / LO_W
                last_seen_t = now

                # Draw bbox on preview
                cv2.rectangle(dbg, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), (255,255,255), 2)

            # Debug inset
            inset = cv2.resize(pooled, (160, 120))
            dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

        prev_lo = lo_y

        # ---- Track x over short time window for span + velocity ----
        # prune old points
        while trace and (now - trace[0][0]) > SPAN_WINDOW_S:
            trace.popleft()

        if cx_norm is not None:
            trace.append((now, cx_norm))

        # compute span and velocity sign
        span = 0.0
        vel  = 0.0
        dir_sign = 0
        if len(trace) >= 2:
            xs = [p[1] for p in trace]
            ts = [p[0] for p in trace]
            span = max(xs) - min(xs)
            # instantaneous velocity from last two points
            dt = max(1e-3, ts[-1] - ts[-2])
            vel = (xs[-1] - xs[-2]) / dt
            dir_sign = 1 if vel > 0 else (-1 if vel < 0 else 0)

        # ---- State machine + span/velocity firing ----
        gesture_text = ""
        can_fire = (now - last_fire) > COOLDOWN_S

        if cx_norm is not None:
            if state == "IDLE":
                if cx_norm <= L_ARM:
                    state = "ARM_RIGHT"
                elif cx_norm >= R_ARM:
                    state = "ARM_LEFT"

            # fire by gates
            if can_fire:
                if state == "ARM_RIGHT" and cx_norm >= R_FIRE:
                    print("SWIPE_RIGHT")
                    gesture_text = "SWIPE_RIGHT"
                    last_fire = now
                    state = "IDLE"
                elif state == "ARM_LEFT" and cx_norm <= L_FIRE:
                    print("SWIPE_LEFT")
                    gesture_text = "SWIPE_LEFT"
                    last_fire = now
                    state = "IDLE"

            # fire by span+velocity (mid-screen flicks)
            if can_fire and gesture_text == "" and span >= SPAN_THR and abs(vel) >= VEL_THR:
                if dir_sign > 0:
                    print("SWIPE_RIGHT")
                    gesture_text = "SWIPE_RIGHT"
                elif dir_sign < 0:
                    print("SWIPE_LEFT")
                    gesture_text = "SWIPE_LEFT"
                if gesture_text:
                    last_fire = now
                    state = "IDLE"
                    trace.clear()  # avoid double-firing
        else:
            if state != "IDLE" and (now - last_seen_t) > ABSENCE_RESET_S:
                state = "IDLE"
            # also clear stale trace to reduce accidental span detections
            if trace and (now - trace[-1][0]) > ABSENCE_RESET_S:
                trace.clear()

        # ---- HUD ----
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.putText(dbg, f"x={trace[-1][1]:.2f}" if trace else "x=--", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(dbg, f"STATE={state}  span={span:.2f}  vel={vel:.2f}w/s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Hand Swipe (ultra fast)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.003)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
