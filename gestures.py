import time, os
import numpy as np
import cv2
from picamera2 import Picamera2

# =========================
# Config
# =========================
FRAME_W, FRAME_H = 640, 480
LO_W, LO_H       = 320, 180

# Crossing gates (must traverse across the full screen)
CROSS_L = 0.20
CROSS_R = 0.80
COOLDOWN_S = 0.20               # quick back-to-back flicks
ABSENCE_RESET_S = 0.25          # if finger vanishes, reset state

# "Huge finger" constraints (in lores coordinates)
MIN_H_FRAC   = 0.92             # >=92% of height
MIN_AR       = 3.8              # tall & skinny
MAX_W_FRAC   = 0.35
MIN_W_FRAC   = 0.02

# Motion / preprocessing
MORPH_K      = 3
ADAPT_K      = 1.8
THR_FLOOR    = 14
HEADLESS     = os.environ.get("DISPLAY", "") == ""

# Hysteresis so tiny jitter around the gates doesn’t flip state
HYST = 0.03
L_ARM  = CROSS_L - HYST
L_FIRE = CROSS_L + HYST
R_ARM  = CROSS_R + HYST
R_FIRE = CROSS_R - HYST

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
    main = {"format":"RGB888", "size": (FRAME_W, FRAME_H)},
    lores= {"format":"YUV420", "size": (LO_W, LO_H)},
    display="main",
)
picam2.configure(config)

try:
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True, "AfMode": 2,
        "FrameDurationLimits": (16666, 16666),   # ~60 fps
        "AnalogueGain": 8.0,
    })
except Exception:
    pass

picam2.start()

# =========================
# State
# =========================
prev_lo = None
morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
last_fire = 0.0

# Swipe state machine
state = "IDLE"                  # IDLE | ARM_RIGHT | ARM_LEFT
last_seen_t = 0.0               # last time we saw a valid blob

def reset_state():
    global state
    state = "IDLE"

try:
    while True:
        now = time.time()

        # ---- Fast detection frame (lores Y) ----
        yuv_lo = picam2.capture_array("lores")
        lo_y   = luma_from_lores(yuv_lo)

        # ---- Preview frame (main) ----
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        cx_norm = None
        bbox_main = None

        # ---- Motion & big-blob filtering ----
        if prev_lo is not None:
            diff = cv2.absdiff(lo_y, prev_lo)
            m, s = cv2.meanStdDev(diff)
            thr = max(THR_FLOOR, float(m[0][0] + ADAPT_K * s[0][0]))
            _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

            th = cv2.medianBlur(th, 3)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph, iterations=1)

            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidate = None
            best_area = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                h_frac = h / LO_H
                w_frac = w / LO_W
                if h_frac < MIN_H_FRAC:
                    continue
                if not (MIN_W_FRAC <= w_frac <= MAX_W_FRAC):
                    continue
                ar = h / max(1, w)
                if ar < MIN_AR:
                    continue
                area = w * h
                if area > best_area:
                    best_area = area
                    candidate = (x, y, w, h)

            if candidate:
                x, y, w, h = candidate
                sx, sy = FRAME_W / LO_W, FRAME_H / LO_H
                bbox_main = (int(x*sx), int(y*sy), int((x+w)*sx), int((y+h)*sy))

                cx_norm = (x + w/2) / LO_W
                last_seen_t = now

        prev_lo = lo_y

        # ---- State machine for guaranteed full crossing ----
        gesture_text = ""
        if cx_norm is not None:
            # Arm conditions
            if state == "IDLE":
                if cx_norm <= L_ARM:
                    state = "ARM_RIGHT"  # started at left, expect move to right
                elif cx_norm >= R_ARM:
                    state = "ARM_LEFT"   # started at right, expect move to left

            # Fire conditions (with cooldown)
            if (now - last_fire) > COOLDOWN_S:
                if state == "ARM_RIGHT" and cx_norm >= R_FIRE:
                    print("SWIPE_RIGHT")
                    gesture_text = "SWIPE_RIGHT"
                    last_fire = now
                    reset_state()
                elif state == "ARM_LEFT" and cx_norm <= L_FIRE:
                    print("SWIPE_LEFT")
                    gesture_text = "SWIPE_LEFT"
                    last_fire = now
                    reset_state()
        else:
            # If absent for a bit, reset the arm (avoid stale arms)
            if state != "IDLE" and (now - last_seen_t) > ABSENCE_RESET_S:
                reset_state()

        # ---- Draw debug ----
        if bbox_main:
            x0, y0, x1, y1 = bbox_main
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (255,255,255), 2)

        # Gates
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)

        # HUD
        cv2.putText(dbg, f"x={cx_norm:.2f}" if cx_norm is not None else "x=--",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(dbg, f"STATE={state}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Finger Swipe (full-cross only)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.005)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
