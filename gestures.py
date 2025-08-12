import time, math, collections, os
import numpy as np
import cv2
from picamera2 import Picamera2
import mediapipe as mp

# ---- Config ----
FRAME_W, FRAME_H = 640, 480
HIST_LEN = 5                  # shorter history => faster swipes
SPEED_MIN = 0.020             # require a bit more motion for swipe
DIR_RATIO = 1.8               # stricter axis dominance
COOLDOWN_S = 0.50             # faster re-trigger for quick flicks
PINCH_RATIO_THR = 0.45

# Motion fallback
DIFF_THRESH = 35              # threshold on absdiff (0-255)
MIN_BLOB_AREA = 300           # ignore tiny noise blobs
MORPH_K = 3                   # morphology kernel size

HEADLESS = os.environ.get("DISPLAY", "") == ""

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---- Camera ----
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"format":"RGB888", "size":(FRAME_W, FRAME_H)}
)
picam2.configure(config)
# target ~30 fps, lower exposure to reduce motion blur if possible
try:
    picam2.set_controls({
        "FrameDurationLimits": (33333, 33333),  # ~30fps
        "ExposureTime": 5000,                   # 1/200s (µs); adjust to taste
        "AnalogueGain": 2.0,
        "AfMode": 2,                            # continuous autofocus if supported
    })
except Exception:
    pass
picam2.start()

# ---- State ----
pts = collections.deque(maxlen=HIST_LEN)   # trajectory (hand or motion)
last_fire = 0
pinch_state = False
prev_gray = None
morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))

def norm(vx, vy): return math.hypot(vx, vy)

def gesture_from_traj(traj):
    if len(traj) < 2: return None
    dx = traj[-1][0] - traj[0][0]
    dy = traj[-1][1] - traj[0][1]
    speed = norm(dx, dy) / max(1, len(traj))
    if speed < SPEED_MIN: return None
    if abs(dx) > DIR_RATIO * abs(dy):
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
    if abs(dy) > DIR_RATIO * abs(dx):
        return "SWIPE_DOWN"  if dy > 0 else "SWIPE_UP"
    return None

def pinch_ratio(lm):
    t, i = lm[4], lm[8]
    w, m = lm[0], lm[9]
    d_tip  = math.hypot(t.x - i.x, t.y - i.y)
    d_size = math.hypot(w.x - m.x, w.y - m.y) + 1e-6
    return d_tip / d_size

try:
    while True:
        frame = picam2.capture_array()        # RGB
        rgb = frame
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        dbg = bgr.copy()

        # ---------- Try hand landmarks first ----------
        res = hands.process(rgb)
        got_hand = False
        gesture_text = ""
        pinch_text = "PINCH: NO"

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            # centroid using a few stable points
            cx = (lm[5].x + lm[9].x + lm[0].x) / 3.0
            cy = (lm[5].y + lm[9].y + lm[0].y) / 3.0
            pts.append((cx, cy))
            got_hand = True

            # Pinch (if user occasionally shows whole hand)
            pr = pinch_ratio(lm)
            is_pinch = pr < PINCH_RATIO_THR
            if is_pinch != pinch_state:
                pinch_state = is_pinch
                print("PINCH_START" if pinch_state else "PINCH_END")
            if pinch_state:
                pinch_text = "PINCH: YES"

        # ---------- Fallback: motion-based finger swipe ----------
        if not got_hand:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, th = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, morph, iterations=1)
                th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, morph, iterations=1)

                cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    c = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(c) >= MIN_BLOB_AREA:
                        (x, y, w, h) = cv2.boundingRect(c)
                        # centroid in normalized coords
                        cx = (x + w/2) / FRAME_W
                        cy = (y + h/2) / FRAME_H
                        pts.append((cx, cy))
                        # debug boxes
                        cv2.rectangle(dbg, (x, y), (x+w, y+h), (255,255,255), 2)
                # optional: show threshold map in a tiny inset
                small = cv2.resize(th, (160, 120))
                dbg[0:120, FRAME_W-160:FRAME_W] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            prev_gray = gray

        # ---------- Decision: swipe (suppressed if pinching) ----------
        now = time.time()
        if (now - last_fire) > COOLDOWN_S and (not pinch_state):
            g = gesture_from_traj(list(pts))
            if g:
                last_fire = now
                gesture_text = g
                print(g)

        # ---------- Draw trajectory ----------
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]*FRAME_W), int(pts[i-1][1]*FRAME_H))
            p2 = (int(pts[i][0]*FRAME_W),   int(pts[i][1]*FRAME_H))
            cv2.line(dbg, p1, p2, (255,255,255), 2)

        # ---------- HUD / show ----------
        cv2.putText(dbg, gesture_text or "", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(dbg, pinch_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        if not HEADLESS:
            cv2.imshow("Gestures (IMX708)", dbg)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            time.sleep(0.01)

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
