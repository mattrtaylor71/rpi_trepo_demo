import time, math, collections, os
import numpy as np
import cv2
from picamera2 import Picamera2
import mediapipe as mp

# ---- Config ----
FRAME_W, FRAME_H = 640, 480
HIST_LEN = 4
SPEED_MIN = 0.022
DIR_RATIO = 2.2
COOLDOWN_S = 0.40
PINCH_RATIO_THR = 0.45

# Finger motion fallback (strict)
DIFF_MIN = 18                 # base threshold floor (auto-thresholded above this)
MIN_AREA = 180                # min contour area to consider
MAX_AREA = 5000               # max contour area (reject big faces/arms)
AR_MIN = 2.5                  # min aspect ratio (long & thin)
AR_MAX = 12.0                 # max aspect ratio
ROI_Y0 = 0.40                 # band start (normalized 0..1)
ROI_Y1 = 0.70                 # band end
FACE_CHECK_EVERY = 4          # run face cascade every N frames

HEADLESS = os.environ.get("DISPLAY", "") == ""

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)

# ---- Face detector (to suppress face motion) ----
FACE_CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# ---- Camera ----
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"format":"RGB888", "size":(FRAME_W, FRAME_H)})
picam2.configure(config)
try:
    picam2.set_controls({"AeEnable": True, "AwbEnable": True, "AfMode": 2})
except Exception:
    pass
picam2.start()

# ---- State ----
pts = collections.deque(maxlen=HIST_LEN)   # normalized (x,y)
last_fire = 0
pinch_state = False
prev_gray = None
frame_idx = 0
roi_px = (int(ROI_Y0*FRAME_H), int(ROI_Y1*FRAME_H))
morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
last_faces = []  # cache faces for N-1 frames

def norm(vx, vy): return math.hypot(vx, vy)

def gesture_from_traj(traj):
    if len(traj) < 2: return None
    x0, y0 = traj[0]
    x1, y1 = traj[-1]
    dx, dy = x1 - x0, y1 - y0
    speed = norm(dx, dy) / max(1, len(traj))
    if speed < SPEED_MIN: return None
    # gate: must traverse across band thirds to count as a swipe
    if dx > 0 and x0 < 0.30 and x1 > 0.70 and abs(dx) > DIR_RATIO*abs(dy):
        return "SWIPE_RIGHT"
    if dx < 0 and x0 > 0.70 and x1 < 0.30 and abs(dx) > DIR_RATIO*abs(dy):
        return "SWIPE_LEFT"
    # (enable vertical gates if you need up/down)
    return None

def pinch_ratio(lm):
    t, i = lm[4], lm[8]
    w, m = lm[0], lm[9]
    d_tip  = math.hypot(t.x - i.x, t.y - i.y)
    d_size = math.hypot(w.x - m.x, w.y - m.y) + 1e-6
    return d_tip / d_size

def overlaps_face(x, y, w, h, faces):
    # simple IoU with any detected face
    a = (x, y, x+w, y+h)
    ax0, ay0, ax1, ay1 = a
    a_area = (ax1-ax0)*(ay1-ay0)
    for (fx, fy, fw, fh) in faces:
        bx0, by0, bx1, by1 = fx, fy, fx+fw, fy+fh
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0, ix1-ix0), max(0, iy1-iy0)
        inter = iw*ih
        if inter > 0 and inter / float(a_area + fw*fh - inter + 1e-6) > 0.15:
            return True
    return False

try:
    while True:
        frame = picam2.capture_array()
        rgb = frame
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        # ---------- Hand landmarks path (preferred) ----------
        res = hands.process(rgb)
        got_hand = False
        gesture_text = ""
        pinch_text = "PINCH: NO"

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            cx = (lm[5].x + lm[9].x + lm[0].x) / 3.0
            cy = (lm[5].y + lm[9].y + lm[0].y) / 3.0
            # only accept if inside ROI band (helps avoid faces)
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

        # ---------- Fallback: finger-like motion in ROI ----------
        faces = last_faces
        if frame_idx % FACE_CHECK_EVERY == 0:
            gray_small = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=3, minSize=(40,40))
            last_faces = faces

        if not got_hand:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Contrast lift
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_eq = clahe.apply(gray)

            if prev_gray is not None:
                diff = cv2.absdiff(gray_eq, prev_gray)
                m, s = cv2.meanStdDev(diff)
                thr = max(DIFF_MIN, float(m[0][0] + 1.8 * s[0][0]))
                _, th = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
                th = cv2.medianBlur(th, 3)
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph, iterations=1)
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph, iterations=1)

                # restrict to ROI band
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
                    (cxp, cyp), (w, h), ang = rect
                    w, h = max(w, 1), max(h, 1)
                    ar = max(w, h) / min(w, h)  # aspect ratio
                    if not (AR_MIN <= ar <= AR_MAX):
                        continue
                    x, y, wbb, hbb = cv2.boundingRect(c)
                    if overlaps_face(x, y, wbb, hbb, faces):
                        continue
                    candidates.append((area, (x, y, wbb, hbb), (cxp, cyp)))

                if candidates:
                    # take biggest valid skinny blob
                    candidates.sort(key=lambda t: t[0], reverse=True)
                    _, (x, y, wbb, hbb), (cxp, cyp) = candidates[0]
                    cx = cxp / FRAME_W
                    cy = cyp / FRAME_H
                    pts.append((cx, cy))
                    # debug draw
                    cv2.rectangle(dbg, (x, y), (x+wbb, y+hbb), (255,255,255), 2)

                # inset view
                inset = cv2.resize(th_masked, (160, 120))
                dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(inset, cv2.COLOR_GRAY2BGR)

            prev_gray = gray_eq

        # ---------- Decide swipe (no pinch) ----------
        now = time.time()
        if (now - last_fire) > COOLDOWN_S and (not pinch_state):
            g = gesture_from_traj(list(pts))
            if g:
                last_fire = now
                gesture_text = g
                print(g)

        # ---------- Draw trajectory & HUD ----------
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]*FRAME_W), int(pts[i-1][1]*FRAME_H))
            p2 = (int(pts[i][0]*FRAME_W),   int(pts[i][1]*FRAME_H))
            cv2.line(dbg, p1, p2, (255,255,255), 2)
        # draw ROI band
        cv2.rectangle(dbg, (0, roi_px[0]), (FRAME_W, roi_px[1]), (255,255,255), 1)

        cv2.putText(dbg, gesture_text or "", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(dbg, pinch_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
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
