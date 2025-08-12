import time, os
import numpy as np
import cv2
from picamera2 import Picamera2

# =========================
# Config (hand swipe mode)
# =========================
FRAME_W, FRAME_H = 640, 480
LO_W, LO_H       = 320, 240         # lores for detection (240 high for vertical coverage)

# Crossing gates (must traverse across most of the screen)
CROSS_L = 0.18
CROSS_R = 0.82
COOLDOWN_S = 0.18                   # quick repeats
ABSENCE_RESET_S = 0.20

# ROI band (wide so whole-hand swipes are seen; adjust to your setup)
ROI_Y0 = 0.22
ROI_Y1 = 0.82

# Hand-like blob constraints (looser than finger mode)
MIN_H_FRAC      = 0.60              # >=60% of lores height
MIN_AREA_FRAC   = 0.10              # >=10% of lores image area
MAX_AREA_FRAC   = 0.85              # ignore near-full-screen blobs
MIN_W_FRAC      = 0.08              # allow a wide hand
MAX_W_FRAC      = 0.85
AR_MIN          = 1.05              # height/width (a bit taller than wide)
SOLIDITY_MIN    = 0.60              # area / convexHullArea

# Motion / preprocessing
ADAPT_K    = 1.6                    # mean + k*std threshold
THR_FLOOR  = 12
MORPH_K    = 3
FACE_EVERY = 4                      # run face cascade every N frames

HEADLESS = os.environ.get("DISPLAY", "") == ""
HYST = 0.02                         # hysteresis around gates
L_ARM, L_FIRE = CROSS_L - HYST, CROSS_L + HYST
R_ARM, R_FIRE = CROSS_R + HYST, CROSS_R - HYST

# =========================
# Helpers
# =========================
def luma_from_lores(yuv_lo):
    # Picamera2 YUV420 lores is planar: Y on top (LO_H x LO_W)
    if yuv_lo.ndim == 2:
        return yuv_lo[:LO_H, :LO_W]
    elif yuv_lo.ndim == 3:
        return yuv_lo[:, :, 0]
    raise ValueError(f"Unexpected lores shape: {yuv_lo.shape}")

def overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
    return iw*ih

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
        "FrameDurationLimits": (16666, 16666),   # ~60 fps target
        "AnalogueGain": 8.0,                     # bump if too dark (4–12 typical)
    })
except Exception:
    pass

picam2.start()

# =========================
# Face suppression
# =========================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# =========================
# State
# =========================
prev_lo       = None
morph         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))
last_fire     = 0.0
state         = "IDLE"            # IDLE | ARM_RIGHT | ARM_LEFT
last_seen_t   = 0.0
frame_idx     = 0
y0_roi, y1_roi = int(ROI_Y0*LO_H), int(ROI_Y1*LO_H)

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

        # ---- Motion & hand-like blob filtering in ROI ----
        if prev_lo is not None:
            diff = cv2.absdiff(lo_y, prev_lo)

            # Adaptive threshold
            m, s = cv2.meanStdDev(diff)
            thr = max(THR_FLOOR, float(m[0][0] + ADAPT_K * s[0][0]))
            _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)

            # Clean up
            th = cv2.medianBlur(th, 3)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph, iterations=1)

            # Mask to ROI band
            mask = np.zeros_like(th)
            mask[y0_roi:y1_roi, :] = 255
            th = cv2.bitwise_and(th, mask)

            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Face suppression (on main, every few frames)
            faces = []
            if frame_idx % FACE_EVERY == 0:
                gray_main = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_main, scaleFactor=1.1, minNeighbors=3, minSize=(64,64))

            candidate = None
            best_area = 0
            lo_area = LO_W * LO_H
            min_area_px = MIN_AREA_FRAC * lo_area
            max_area_px = MAX_AREA_FRAC * lo_area

            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                if y < y0_roi or (y + h) > y1_roi:
                    # must be fully inside ROI band
                    continue

                h_frac = h / LO_H
                w_frac = w / LO_W
                area   = w * h
                if not (MIN_W_FRAC <= w_frac <= MAX_W_FRAC):
                    continue
                if not (min_area_px <= area <= max_area_px):
                    continue
                if h_frac < MIN_H_FRAC:
                    continue

                # Aspect ratio and solidity (hand-ish)
                ar = (h / max(1, w))
                if ar < AR_MIN:
                    continue

                hull = cv2.convexHull(c)
                hull_area = max(cv2.contourArea(hull), 1.0)
                solidity = float(area) / hull_area
                if solidity < SOLIDITY_MIN:
                    continue

                # Map bbox to main coords and reject if overlapping a face
                sx, sy = FRAME_W / LO_W, FRAME_H / LO_H
                x0m, y0m = int(x*sx), int(y*sy)
                x1m, y1m = int((x+w)*sx), int((y+h)*sy)
                reject = False
                for (fx, fy, fw, fh) in faces:
                    if overlaps((x0m,y0m,x1m,y1m), (fx,fy,fx+fw,fy+fh)) > 0:
                        reject = True
                        break
                if reject:
                    continue

                if area > best_area:
                    best_area = area
                    candidate = (x, y, w, h, x0m, y0m, x1m, y1m)

            if candidate:
                x, y, w, h, x0m, y0m, x1m, y1m = candidate
                cx_norm = (x + w/2) / LO_W
                last_seen_t = now
                bbox_main = (x0m, y0m, x1m, y1m)

            # Debug inset
            inset = cv2.resize(th, (160, 120))
            dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

        prev_lo = lo_y

        # ---- State machine: require full crossing ----
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

        # ---- Draw debug ----
        if bbox_main:
            x0, y0, x1, y1 = bbox_main
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (255,255,255), 2)

        # ROI & gates
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
            cv2.imshow("Hand Swipe (full-cross)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.005)

        frame_idx += 1

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
