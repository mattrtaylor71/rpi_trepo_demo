import time, math, collections, os
import numpy as np
import cv2
from picamera2 import Picamera2
import mediapipe as mp

# =========================
# Config
# =========================
FRAME_W, FRAME_H = 640, 480      # preview
LO_W, LO_H       = 320, 180      # faster lores (use 240 if you prefer)

# Swipe logic
HIST_LEN        = 4
SPEED_MIN       = 0.020
DIR_RATIO       = 2.4
COOLDOWN_S      = 0.30
PINCH_RATIO_THR = 0.45

# Finger-only motion (strict + ROI)
ROI_Y0       = 0.56     # tighten the band to where the finger will pass
ROI_Y1       = 0.72
CROSS_L      = 0.28
CROSS_R      = 0.72
MIN_AREA     = 120      # px^2 in lores
MAX_AREA     = 2200
AR_MIN       = 3.0      # long & thin
AR_MAX       = 18.0
ECC_MIN      = 0.92     # 0..1 (1 = line). Enforces elongation.
MOG_HISTORY  = 40
MOG_VAR_THR  = 16
MOG_WARMUP   = 15       # frames to learn background

HEADLESS = os.environ.get("DISPLAY", "") == ""

# =========================
# Helpers
# =========================
def norm(vx, vy):
    return math.hypot(vx, vy)

def gesture_from_traj(traj):
    if len(traj) < 2: return None
    x0, y0 = traj[0]; x1, y1 = traj[-1]
    dx, dy = x1 - x0, y1 - y0
    speed  = norm(dx, dy) / max(1, len(traj))
    if speed < SPEED_MIN: return None
    if abs(dx) > DIR_RATIO * abs(dy): return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
    if abs(dy) > DIR_RATIO * abs(dx): return "SWIPE_DOWN"  if dy > 0 else "SWIPE_UP"
    return None

def crossing_swipe(x_prev, x_curr):
    if x_prev is None or x_curr is None: return None
    if x_prev < CROSS_L and x_curr > CROSS_R: return "SWIPE_RIGHT"
    if x_prev > CROSS_R and x_curr < CROSS_L: return "SWIPE_LEFT"
    return None

def pinch_ratio(lm):
    t, i = lm[4], lm[8]
    w, m = lm[0], lm[9]
    d_tip  = math.hypot(t.x - i.x, t.y - i.y)
    d_size = math.hypot(w.x - m.x, w.y - m.y) + 1e-6
    return d_tip / d_size

def luma_from_lores(yuv_lo):
    # planar YUV420: take Y plane (top LO_H rows)
    if yuv_lo.ndim == 2:
        return yuv_lo[:LO_H, :LO_W]
    elif yuv_lo.ndim == 3:
        return yuv_lo[:, :, 0]
    raise ValueError(f"Unexpected lores array shape: {yuv_lo.shape}")

def eccentricity(cnt):
    # Eccentricity from covariance eigenvalues of the contour points
    if len(cnt) < 5: return 0.0
    pts = cnt.reshape(-1, 2).astype(np.float32)
    cov = np.cov(pts.T)
    eigvals, _ = np.linalg.eig(cov)
    a, b = np.sqrt(np.maximum(eigvals, 1e-6))  # major/minor-like
    major, minor = (max(a, b), min(a, b))
    if major <= 1e-6: return 0.0
    e = math.sqrt(1.0 - (minor*minor)/(major*major))
    return float(np.clip(e, 0.0, 0.9999))

# =========================
# MediaPipe Hands (pinch + centroid when full hand visible)
# =========================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.70,
    min_tracking_confidence=0.70,
)

# Face suppression (portable path)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

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
    picam2.set_controls({
        "AeEnable": True,
        "AwbEnable": True,
        "AfMode": 2,
        "FrameDurationLimits": (16666, 16666),   # ~60 fps
        "AnalogueGain": 8.0,                     # bump if dark (4–12)
    })
except Exception:
    pass

picam2.start()

# =========================
# State & detectors
# =========================
pts           = collections.deque(maxlen=HIST_LEN)  # normalized history
last_fire     = 0.0
pinch_state   = False
prev_lo_eq    = None
frame_idx     = 0
roi_px        = (int(ROI_Y0*LO_H), int(ROI_Y1*LO_H))  # lores ROI
last_x_norm   = None

mog = cv2.createBackgroundSubtractorMOG2(history=MOG_HISTORY, varThreshold=MOG_VAR_THR, detectShadows=False)
morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

try:
    while True:
        # -------- Fast detection frame (lores Y) --------
        yuv_lo  = picam2.capture_array("lores")
        lo_y    = luma_from_lores(yuv_lo)

        # -------- Pretty preview (main) --------
        rgb = picam2.capture_array("main")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        # -------- Hand landmarks (preferred) --------
        got_hand = False
        gesture_text = ""
        pinch_text   = "PINCH: NO"

        res = hands.process(rgb)  # MP needs RGB
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            cx = (lm[5].x + lm[9].x + lm[0].x) / 3.0
            cy = (lm[5].y + lm[9].y + lm[0].y) / 3.0
            if ROI_Y0 <= cy <= ROI_Y1:
                pts.append((cx, cy))
                got_hand = True

            pr = pinch_ratio(lm)
            is_pinch = pr < PINCH_RATIO_THR
            if is_pinch != pinch_state:
                pinch_state = is_pinch
                print("PINCH_START" if pinch_state else "PINCH_END")
            if pinch_state:
                pinch_text = "PINCH: YES"

        # -------- Finger-only fallback on lores (very responsive) --------
        cx_norm_for_cross = None
        if not got_hand:
            # Light equalization for stability
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lo_eq = clahe.apply(lo_y)

            # Warmup background model
            lr = 0.01 if frame_idx > MOG_WARMUP else 0.5
            fg = mog.apply(lo_eq, learningRate=lr)  # 0..255

            # ROI mask
            y0, y1 = roi_px
            mask = np.zeros_like(fg)
            mask[y0:y1, :] = 255
            fg = cv2.bitwise_and(fg, mask)

            # Clean up
            fg = cv2.medianBlur(fg, 3)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph, iterations=1)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, morph, iterations=1)

            # Find skinny finger-like blobs
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Face suppression: run on main (scaled down)
            faces = []
            if frame_idx % 4 == 0:
                gray_main = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_main, scaleFactor=1.1, minNeighbors=3, minSize=(64,64))

            candidates = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < MIN_AREA or area > MAX_AREA:
                    continue

                # Geometry checks
                rect = cv2.minAreaRect(c)
                (cxp, cyp), (w, h), _ = rect
                w, h = max(w, 1), max(h, 1)
                ar = max(w, h) / min(w, h)
                if not (AR_MIN <= ar <= AR_MAX):
                    continue

                # Eccentricity (elongation)
                ecc = eccentricity(c)
                if ecc < ECC_MIN:
                    continue

                # Map bbox to main coords to avoid faces
                x, y, wbb, hbb = cv2.boundingRect(c)
                x0m = int(x * (FRAME_W / LO_W))
                y0m = int(y * (FRAME_H / LO_H))
                x1m = int((x + wbb) * (FRAME_W / LO_W))
                y1m = int((y + hbb) * (FRAME_H / LO_H))

                # Suppress if overlapping any face bbox
                overlaps_face = False
                for (fx, fy, fw, fh) in faces:
                    ix0, iy0 = max(x0m, fx), max(y0m, fy)
                    ix1, iy1 = min(x1m, fx+fw), min(y1m, fy+fh)
                    if ix1 > ix0 and iy1 > iy0:
                        overlaps_face = True
                        break
                if overlaps_face:
                    continue

                candidates.append((area, (x, y, wbb, hbb), (cxp, cyp)))

            if candidates:
                candidates.sort(key=lambda t: t[0], reverse=True)
                _, (x, y, wbb, hbb), (cxp, cyp) = candidates[0]

                # Normalized lores coords
                cx_norm = cxp / LO_W
                cy_norm = cyp / LO_H
                pts.append((cx_norm, cy_norm))
                cx_norm_for_cross = cx_norm

                # Draw on preview
                x0m = int(x * (FRAME_W / LO_W))
                y0m = int(y * (FRAME_H / LO_H))
                x1m = int((x + wbb) * (FRAME_W / LO_W))
                y1m = int((y + hbb) * (FRAME_H / LO_H))
                cv2.rectangle(dbg, (x0m, y0m), (x1m, y1m), (255, 255, 255), 2)

            # Debug inset
            inset = cv2.resize(fg, (160, 120))
            dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

        # -------- Decide swipe (crossing gate first, then trajectory) --------
        now = time.time()
        fired = False

        if cx_norm_for_cross is not None and (now - last_fire) > COOLDOWN_S and not pinch_state:
            sw = crossing_swipe(last_x_norm, cx_norm_for_cross)
            last_x_norm = cx_norm_for_cross
            if sw:
                last_fire = now
                gesture_text = sw
                print(sw)
                fired = True
        else:
            # keep last_x_norm updated when hand centroid available
            if pts:
                last_x_norm = pts[-1][0]

        if not fired and (now - last_fire) > COOLDOWN_S and not pinch_state:
            g = gesture_from_traj(list(pts))
            if g:
                last_fire = now
                gesture_text = g
                print(g)
                fired = True

        # -------- Draw trajectory & HUD --------
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]*FRAME_W), int(pts[i-1][1]*FRAME_H))
            p2 = (int(pts[i][0]*FRAME_W),   int(pts[i][1]*FRAME_H))
            cv2.line(dbg, p1, p2, (255,255,255), 2)

        # ROI band + crossing gates
        roi_y0_main = int(ROI_Y0 * FRAME_H)
        roi_y1_main = int(ROI_Y1 * FRAME_H)
        cv2.rectangle(dbg, (0, roi_y0_main), (FRAME_W, roi_y1_main), (255,255,255), 1)
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
