import time, math, collections, os
import numpy as np
import cv2
from picamera2 import Picamera2
import mediapipe as mp

# =========================
# Config
# =========================
FRAME_W, FRAME_H = 640, 480      # preview window
LO_W, LO_H       = 320, 240      # fast detection stream

# Swipe logic
HIST_LEN        = 4
SPEED_MIN       = 0.022
DIR_RATIO       = 2.2
COOLDOWN_S      = 0.40
PINCH_RATIO_THR = 0.45

# Finger-like motion fallback (strict & ROI-limited)
DIFF_MIN     = 18     # base threshold floor; auto-threshold adds to this
MIN_AREA     = 180    # min contour area
MAX_AREA     = 5000   # max contour area
AR_MIN       = 2.5    # min aspect ratio (long & thin)
AR_MAX       = 12.0   # max aspect ratio
ROI_Y0       = 0.40   # detection band (normalized 0..1 of full frame height)
ROI_Y1       = 0.70
CROSS_L      = 0.30   # crossing gate left (normalized)
CROSS_R      = 0.70   # crossing gate right

HEADLESS = os.environ.get("DISPLAY", "") == ""

# =========================
# Helpers
# =========================
def norm(vx, vy): 
    return math.hypot(vx, vy)

def gesture_from_traj(traj):
    """Classic swipe from accumulated points (normalized coords, 0..1)."""
    if len(traj) < 2: 
        return None
    x0, y0 = traj[0]
    x1, y1 = traj[-1]
    dx, dy = x1 - x0, y1 - y0
    speed  = norm(dx, dy) / max(1, len(traj))
    if speed < SPEED_MIN:
        return None
    if abs(dx) > DIR_RATIO * abs(dy):
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
    if abs(dy) > DIR_RATIO * abs(dx):
        return "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"
    return None

def crossing_swipe(x_prev, x_curr):
    """Fire even for 1–2 frame flicks if it crosses gates."""
    if x_prev is None or x_curr is None:
        return None
    if x_prev < CROSS_L and x_curr > CROSS_R:
        return "SWIPE_RIGHT"
    if x_prev > CROSS_R and x_curr < CROSS_L:
        return "SWIPE_LEFT"
    return None

def pinch_ratio(lm):
    t, i = lm[4], lm[8]
    w, m = lm[0], lm[9]
    d_tip  = math.hypot(t.x - i.x, t.y - i.y)
    d_size = math.hypot(w.x - m.x, w.y - m.y) + 1e-6
    return d_tip / d_size

# =========================
# MediaPipe Hands (for pinch + stable centroid when full hand is visible)
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)

# =========================
# Camera: dual-stream (main for preview, lores for fast detection)
# =========================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format":"RGB888", "size": (FRAME_W, FRAME_H)},
    lores={"format":"YUV420", "size": (LO_W, LO_H)},
    display="main",  # show main stream in preview
)
picam2.configure(config)

# Aim for high FPS; keep AE on so it’s not black. Adjust gain for brightness.
try:
    picam2.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": 2,                         # continuous AF if supported
        "FrameDurationLimits": (16666, 16666),  # ~60 fps target
        "AnalogueGain": 8.0,                 # bump if too dark (4–12 typical)
    })
except Exception:
    pass

picam2.start()

# =========================
# State
# =========================
pts         = collections.deque(maxlen=HIST_LEN)  # normalized points history
last_fire   = 0.0
pinch_state = False
prev_gray_lo = None
frame_idx   = 0
roi_px      = (int(ROI_Y0*LO_H), int(ROI_Y1*LO_H))  # ROI in lores coords
morph       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
last_x_norm = None  # for crossing gate

try:
    while True:
        # -------- Fast detection frame (lores) --------
        yuv_lo = picam2.capture_array("lores")     # YUV420
        lo_y   = yuv_lo[:, :, 0]                   # luma plane (LO_H x LO_W)

        # -------- Pretty preview frame (main) --------
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        # -------- Hand landmarks (preferred when full hand is visible) --------
        got_hand = False
        gesture_text = ""
        pinch_text   = "PINCH: NO"

        res = hands.process(rgb)  # MediaPipe expects RGB
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            cx = (lm[5].x + lm[9].x + lm[0].x) / 3.0
            cy = (lm[5].y + lm[9].y + lm[0].y) / 3.0
            # Only accept hand centroid if it lies within the (full-frame) ROI band
            # Map ROI from lores to main normalized (same 0..1 range, so reuse)
            if ROI_Y0 <= cy <= ROI_Y1:
                pts.append((cx, cy))
                got_hand = True

            # Pinch detection
            pr = pinch_ratio(lm)
            is_pinch = pr < PINCH_RATIO_THR
            if is_pinch != pinch_state:
                pinch_state = is_pinch
                print("PINCH_START" if pinch_state else "PINCH_END")
            if pinch_state:
                pinch_text = "PINCH: YES"

        # -------- Fallback: finger-like motion in lores ROI --------
        cx_norm_for_cross = None
        if not got_hand:
            # Contrast boost (CLAHE) on lores grayscale
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lo_eq = clahe.apply(lo_y)

            if prev_gray_lo is not None:
                diff = cv2.absdiff(lo_eq, prev_gray_lo)
                m, s = cv2.meanStdDev(diff)
                thr = max(DIFF_MIN, float(m[0][0] + 1.8 * s[0][0]))
                _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
                th = cv2.medianBlur(th, 3)
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph, iterations=1)
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph, iterations=1)

                # Restrict to vertical ROI band in lores space
                th_masked = np.zeros_like(th)
                y0, y1 = roi_px
                th_masked[y0:y1, :] = th[y0:y1, :]

                cnts, _ = cv2.findContours(th_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                candidates = []
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < MIN_AREA or area > MAX_AREA:
                        continue
                    rect = cv2.minAreaRect(c)
                    (cxp, cyp), (w, h), _ = rect
                    w, h = max(w, 1), max(h, 1)
                    ar = max(w, h) / min(w, h)
                    if not (AR_MIN <= ar <= AR_MAX):
                        continue
                    x, y, wbb, hbb = cv2.boundingRect(c)
                    candidates.append((area, (x, y, wbb, hbb), (cxp, cyp)))

                if candidates:
                    # pick largest skinny blob
                    candidates.sort(key=lambda t: t[0], reverse=True)
                    _, (x, y, wbb, hbb), (cxp, cyp) = candidates[0]

                    # Normalized coords (lores -> 0..1)
                    cx_norm = cxp / LO_W
                    cy_norm = cyp / LO_H
                    pts.append((cx_norm, cy_norm))
                    cx_norm_for_cross = cx_norm

                    # Draw bbox on preview (scale to main)
                    x0 = int(x * (FRAME_W / LO_W))
                    y0 = int(y * (FRAME_H / LO_H))
                    x1 = int((x + wbb) * (FRAME_W / LO_W))
                    y1 = int((y + hbb) * (FRAME_H / LO_H))
                    cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 255, 255), 2)

                # Inset: threshold map (scaled) for debugging
                inset = cv2.resize(th_masked, (160, 120))
                dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

            prev_gray_lo = lo_eq

        # -------- Decide swipe (cooldown + crossing gate + classic traj) --------
        now = time.time()
        fired = False

        # Crossing gate (robust to 1–2 frame flicks)
        if cx_norm_for_cross is not None and (now - last_fire) > COOLDOWN_S and not pinch_state:
            sw = crossing_swipe(last_x_norm, cx_norm_for_cross)
            last_x_norm = cx_norm_for_cross
            if sw:
                last_fire = now
                gesture_text = sw
                print(sw)
                fired = True
        else:
            # keep last_x_norm fresh when MediaPipe hand centroid is available
            if got_hand:
                last_x_norm = pts[-1][0] if pts else last_x_norm

        # Classic trajectory swipe (if not already fired)
        if not fired and (now - last_fire) > COOLDOWN_S and not pinch_state:
            g = gesture_from_traj(list(pts))
            if g:
                last_fire = now
                gesture_text = g
                print(g)
                fired = True

        # -------- Draw trajectory & HUD on preview --------
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]*FRAME_W), int(pts[i-1][1]*FRAME_H))
            p2 = (int(pts[i][0]*FRAME_W),   int(pts[i][1]*FRAME_H))
            cv2.line(dbg, p1, p2, (255,255,255), 2)

        # ROI band (on preview)
        roi_y0_main = int(ROI_Y0 * FRAME_H)
        roi_y1_main = int(ROI_Y1 * FRAME_H)
        cv2.rectangle(dbg, (0, roi_y0_main), (FRAME_W, roi_y1_main), (255,255,255), 1)

        # Crossing gates (visual)
        cv2.line(dbg, (int(CROSS_L*FRAME_W), 0), (int(CROSS_L*FRAME_W), FRAME_H), (255,255,255), 1)
        cv2.line(dbg, (int(CROSS_R*FRAME_W), 0), (int(CROSS_R*FRAME_W), FRAME_H), (255,255,255), 1)

        cv2.putText(dbg, gesture_text if 'gesture_text' in locals() else "", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(dbg, "PINCH: YES" if pinch_state else "PINCH: NO", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if not HEADLESS:
            cv2.imshow("Gestures (IMX708)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.01)

        frame_idx += 1

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
