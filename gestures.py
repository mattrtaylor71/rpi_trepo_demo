import time, os
import numpy as np
import cv2
from picamera2 import Picamera2

# =========================
# Config (loose hand swipe)
# =========================
FRAME_W, FRAME_H = 640, 480
LO_W, LO_H       = 320, 240            # keep decent vertical coverage

# Crossing gates (easier to hit)
CROSS_L = 0.12
CROSS_R = 0.88
COOLDOWN_S = 0.16
ABSENCE_RESET_S = 0.20
HYST = 0.015
L_ARM, L_FIRE = CROSS_L - HYST, CROSS_L + HYST
R_ARM, R_FIRE = CROSS_R + HYST, CROSS_R - HYST

# ROI band (wide) — blob only needs to overlap the band, not be fully inside
ROI_Y0 = 0.15
ROI_Y1 = 0.90
MIN_ROI_OVERLAP_FRAC = 0.60    # at least 60% of blob height inside the band

# Hand-like blob constraints (very loose)
MIN_H_FRAC      = 0.45         # >=45% of lores height
MIN_AREA_FRAC   = 0.05         # >=5% of lores image area
MAX_AREA_FRAC   = 0.92         # ignore near-full-screen blobs
MIN_W_FRAC      = 0.06         # allow wide hand + blur
MAX_W_FRAC      = 0.95
AR_MIN          = 0.80         # allow almost square with blur

# Motion / preprocessing (more sensitive, a bit more merging)
ADAPT_K   = 1.4                # mean + k*std threshold
THR_FLOOR = 10
MORPH_OPEN = 1                 # noise clean
MORPH_CLOSE = 2                # fill small gaps
MORPH_DILATE = 1               # merge finger + palm
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

HEADLESS = os.environ.get("DISPLAY", "") == ""

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

def roi_overlap_fraction(y, h, y0_roi, y1_roi):
    """Return fraction of blob height that lies inside ROI band (0..1)."""
    top, bot = y, y + h
    inter = max(0, min(bot, y1_roi) - max(top, y0_roi))
    return inter / max(1, h)

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
        "FrameDurationLimits": (16666, 16666),   # ~60 fps (drop to 30–45 if too dark)
        "AnalogueGain": 10.0,                    # brighten to help motion
    })
except Exception:
    pass

picam2.start()

# =========================
# State
# =========================
prev_lo      = None
last_fire    = 0.0
state        = "IDLE"   # IDLE | ARM_RIGHT | ARM_LEFT
last_seen_t  = 0.0

y0_roi, y1_roi = int(ROI_Y0*LO_H), int(ROI_Y1*LO_H)
sx, sy = FRAME_W / LO_W, FRAME_H / LO_H

try:
    while True:
        now = time.time()

        # ---- Fast detection frame (lores Y) ----
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
            m, s = cv2.meanStdDev(diff)
            thr = max(THR_FLOOR, float(m[0][0] + ADAPT_K * s[0][0]))
            _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

            # Morphology: open -> close -> dilate
            if MORPH_OPEN:  th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  KERNEL, iterations=MORPH_OPEN)
            if MORPH_CLOSE: th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, KERNEL, iterations=MORPH_CLOSE)
            if MORPH_DILATE:th = cv2.dilate(th, KERNEL, iterations=MORPH_DILATE)

            # Find candidates
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lo_area = LO_W * LO_H
            min_area_px = MIN_AREA_FRAC * lo_area
            max_area_px = MAX_AREA_FRAC * lo_area

            candidate = None
            best_area = 0
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                h_frac = h / LO_H
                w_frac = w / LO_W
                if not (min_area_px <= area <= max_area_px):
                    continue
                if h_frac < MIN_H_FRAC:
                    continue
                if not (MIN_W_FRAC <= w_frac <= MAX_W_FRAC):
                    continue
                # Allow blur: low AR_MIN
                ar = h / max(1, w)
                if ar < AR_MIN:
                    continue
                # Require that most of the blob lies inside ROI band
                if roi_overlap_fraction(y, h, y0_roi, y1_roi) < MIN_ROI_OVERLAP_FRAC:
                    continue

                if area > best_area:
                    best_area = area
                    candidate = (x, y, w, h)

            if candidate:
                x, y, w, h = candidate
                cx_norm = (x + w/2) / LO_W
                last_seen_t = now
                # Draw bbox on preview
                x0m, y0m = int(x*sx), int(y*sy)
                x1m, y1m = int((x+w)*sx), int((y+h)*sy)
                bbox_main = (x0m, y0m, x1m, y1m)

            # Debug inset
            inset = cv2.resize(th, (160, 120))
            dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

        prev_lo = lo_y

        # ---- State machine: require full crossing left<->right ----
        gesture_text = ""
        if cx_norm is not None:
            if state == "IDLE":
                if cx_norm <= L_ARM:
                    state = "ARM_RIGHT"
                elif cx_norm >= R_ARM:
                    state = "ARM_LEFT"

            if (now - last_fire) > COOLDOWN_S:
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
        else:
            if state != "IDLE" and (now - last_seen_t) > ABSENCE_RESET_S:
                state = "IDLE"

        # ---- Draw ----
        if bbox_main:
            x0, y0, x1, y1 = bbox_main
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (255,255,255), 2)

        # ROI band + crossing gates
        cv2.rectangle(dbg, (0, int(ROI_Y0*FRAME_H)), (FRAME_W, int(ROI_Y1*FRAME_H)), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)

        cv2.putText(dbg, f"x={cx_norm:.2f}" if cx_norm is not None else "x=--",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(dbg, f"STATE={state}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if gesture_text:
            cv2.putText(dbg, gesture_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Hand Swipe (loose, full-cross)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.005)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
