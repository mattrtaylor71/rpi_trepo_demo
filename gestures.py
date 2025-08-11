import time, math, collections
import numpy as np
import cv2
from picamera2 import Picamera2
import mediapipe as mp

# ---- Config ----
FRAME_W, FRAME_H = 640, 480
HIST_LEN = 7                 # frames to consider for motion
SPEED_MIN = 0.015            # min normalized speed to count as a swipe
DIR_RATIO = 1.7              # dominant axis multiplier (e.g., |dx| > 1.7*|dy|)
COOLDOWN_S = 0.70            # gesture cooldown to avoid repeats
PINCH_RATIO_THR = 0.45       # thumb-index distance / hand size

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,       # light & fast on Pi
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ---- Camera ----
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format":"RGB888", "size":(FRAME_W, FRAME_H)})
picam2.configure(config)
picam2.start()

pts = collections.deque(maxlen=HIST_LEN)
last_fire = 0
pinch_state = False

def norm(vx, vy):
    return math.sqrt(vx*vx + vy*vy)

def gesture_from_traj(traj):
    # traj is list of (x,y) in [0..1] normalized coords over last frames
    if len(traj) < 2: return None
    dx = traj[-1][0] - traj[0][0]
    dy = traj[-1][1] - traj[0][1]
    speed = norm(dx, dy) / max(1, len(traj))
    if speed < SPEED_MIN: return None

    # dominant axis
    if abs(dx) > DIR_RATIO * abs(dy):
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
    if abs(dy) > DIR_RATIO * abs(dx):
        return "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"
    return None

def pinch_ratio(lm):
    # lm: list of 21 landmarks (normalized)
    t = lm[4]   # thumb tip
    i = lm[8]   # index tip
    w = lm[0]   # wrist
    m = lm[9]   # middle mcp (a rough scale reference)

    d_tip = math.hypot(t.x - i.x, t.y - i.y)
    d_size = math.hypot(w.x - m.x, w.y - m.y) + 1e-6
    return d_tip / d_size

try:
    while True:
        frame = picam2.capture_array()        # RGB
        rgb = frame
        res = hands.process(rgb)

        gesture_text = ""
        pinch_text = "PINCH: NO"

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark

            # centroid of key points (index MCP, middle MCP, wrist) for stability
            cx = (lm[5].x + lm[9].x + lm[0].x) / 3.0
            cy = (lm[5].y + lm[9].y + lm[0].y) / 3.0
            pts.append((cx, cy))

            # pinch detection
            pr = pinch_ratio(lm)
            is_pinch = pr < PINCH_RATIO_THR
            if is_pinch != pinch_state:
                pinch_state = is_pinch
                if pinch_state:
                    print("PINCH_START")
                else:
                    print("PINCH_END")
            if pinch_state:
                pinch_text = "PINCH: YES"

            # swipe detection (only when not pinching)
            now = time.time()
            if not pinch_state and (now - last_fire) > COOLDOWN_S:
                g = gesture_from_traj(list(pts))
                if g:
                    last_fire = now
                    gesture_text = g
                    print(g)

            # draw a tiny trail for debugging
            dbg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][0]*FRAME_W), int(pts[i-1][1]*FRAME_H))
                p2 = (int(pts[i][0]*FRAME_W),   int(pts[i][1]*FRAME_H))
                cv2.line(dbg, p1, p2, (255, 255, 255), 2)
        else:
            dbg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pts.clear()
            pinch_state = False

        # HUD
        cv2.putText(dbg, gesture_text or "", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(dbg, pinch_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.imshow("Gestures (IMX708)", dbg)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
