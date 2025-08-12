import time, math, collections, os
import numpy as np
import cv2
from picamera2 import Picamera2

# =========================
# Config
# =========================
FRAME_W, FRAME_H = 640, 480      # preview
LO_W, LO_H       = 320, 180      # fast detection stream (height matters most)

# Crossing gates (must traverse across the full screen)
CROSS_L = 0.20
CROSS_R = 0.80
COOLDOWN_S = 0.25                # allow quick successive swipes

# "Huge finger" constraints (in lores coordinates)
MIN_H_FRAC   = 0.90              # blob must be >= 90% of frame height
MIN_AR       = 3.5               # height/width must be tall & skinny
MAX_W_FRAC   = 0.40              # safety: reject extremely wide blobs
MIN_W_FRAC   = 0.02              # but not razor-thin noise

# Motion / preprocessing
MORPH_K      = 3
ADAPT_K      = 1.8               # mean + k*std threshold
THR_FLOOR    = 14                # never below this threshold
HEADLESS     = os.environ.get("DISPLAY", "") == ""

# =========================
# Helpers
# =========================
def luma_from_lores(yuv_lo):
    # Picamera2 YUV420 lores: planar; Y on top (LO_H x LO_W)
    if yuv_lo.ndim == 2:
        return yuv_lo[:LO_H, :LO_W]
    elif yuv_lo.ndim == 3:
        return yuv_lo[:, :, 0]
    raise ValueError(f"Unexpected lores shape: {yuv_lo.shape}")

def crossing_swipe(x_prev, x_curr):
    if x_prev is None or x_curr is None:
        return None
    if x_prev < CROSS_L and x_curr > CROSS_R:
        return "SWIPE_RIGHT"
    if x_prev > CROSS_R and x_curr < CROSS_L:
        return "SWIPE_LEFT"
    return None

# =========================
# Camera: dual-stream
# =========================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main = {"format":"RGB888", "size": (FRAME_W, FRAME_H)},  # pretty preview
    lores= {"format":"YUV420", "size": (LO_W, LO_H)},        # fast detection
    display="main",
)
picam2.configure(config)

try:
    picam2.set_controls({
        "AeEnable": True, "AwbEnable": True,
        "AfMode": 2,                               # continuous AF if supported
        "FrameDurationLimits": (16666, 16666),     # ~60 fps target
        "AnalogueGain": 8.0,                       # bump if too dark (4–12)
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
last_x_norm = None

try:
    while True:
        # ---- Fast detection frame (lores Y) ----
        yuv_lo = picam2.capture_array("lores")
        lo_y   = luma_from_lores(yuv_lo)

        # ---- Preview frame (main) ----
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        cx_norm_for_cross = None

        # ---- Motion & big-blob filtering ----
        if prev_lo is not None:
            diff = cv2.absdiff(lo_y, prev_lo)

            # Adaptive threshold: mean + k*std, floored
            m, s = cv2.meanStdDev(diff)
            thr = max(THR_FLOOR, float(m[0][0] + ADAPT_K * s[0][0]))
            _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

            # Clean up
            th = cv2.medianBlur(th, 3)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph, iterations=1)

            # Contours
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            candidate = None
            best_area = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                h_frac = h / LO_H
                w_frac = w / LO_W
                if h_frac < MIN_H_FRAC:     # must span nearly full height
                    continue
                if w_frac > MAX_W_FRAC or w_frac < MIN_W_FRAC:
                    continue
                ar = (h / max(1, w))        # tall & skinny
                if ar < MIN_AR:
                    continue

                area = w * h
                if area > best_area:
                    best_area = area
                    candidate = (x, y, w, h)

            if candidate:
                x, y, w, h = candidate
                # centroid normalized to 0..1 (lores -> normalized)
                cx_norm = (x + w/2) / LO_W
                cx_norm_for_cross = cx_norm

                # draw bbox on preview (scale coords)
                sx = FRAME_W / LO_W
                sy = FRAME_H / LO_H
                cv2.rectangle(dbg,
                              (int(x*sx), int(y*sy)),
                              (int((x+w)*sx), int((y+h)*sy)),
                              (255,255,255), 2)

            # Debug inset
            inset = cv2.resize(th, (160, 120))
            dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

        prev_lo = lo_y

        # ---- Fire only on full crossing of the screen ----
        now = time.time()
        gesture_text = ""
        if cx_norm_for_cross is not None and (now - last_fire) > COOLDOWN_S:
            sw = crossing_swipe(last_x_norm, cx_norm_for_cross)
            last_x_norm = cx_norm_for_cross
            if sw:
                last_fire = now
                gesture_text = sw
                print(sw)

        # ---- Visual gates ----
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.putText(dbg, gesture_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Finger Swipe (huge-blob only)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.005)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
