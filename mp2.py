import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

# =========================
#MADE BY HIMANSHU RAJ
# Configuration / tuning
# =========================
pyautogui.FAILSAFE = False
SMOOTH_WINDOW = 6            # moving average window for cursor
EXP_SMOOTH = 0.6             # exponential smoothing factor (0-1) lower = smoother/slower

PINCH_ENTER = 0.05           # pinch enter threshold (normalized distance)
PINCH_EXIT = 0.08            # pinch exit threshold (hysteresis)
DOUBLE_PINCH_TIME = 0.45     # seconds for double pinch (double-click)

# Scrolling (while pinched)
SCROLL_SENSITIVITY = 1200.0  # larger => more scroll per normalized move
SCROLL_DEADZONE = 0.008      # ignore tiny vertical moves while scrolling

HAND_STABLE_FRAMES = 3       # require this many consecutive hand frames to enable control
MIRROR = True                # mirror webcam frame for display (keep intuitive left/right)

# Screen coverage tuning: values >1.0 expand movement range so small hand motions cover full screen
HORIZONTAL_SCALE = 1.6
VERTICAL_SCALE = 1.3

# Fist click tuning
FIST_DEBOUNCE = 0.6         # seconds between fist clicks

# Blink-to-scroll tuning
BLINK_THRESHOLD = 0.15      # EAR threshold (lower = more sensitive)
BLINK_DEBOUNCE = 0.7        # seconds between blink-triggered scrolls
BLINK_SCROLL_AMOUNT = -300  # negative scroll = scroll down

# =========================
# Init
# =========================
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
# add FaceMesh for blink detection
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True,
                             max_num_faces=1,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

screen_w, screen_h = pyautogui.size()

# smoothing buffers / state
pos_buf_x = deque(maxlen=SMOOTH_WINDOW)
pos_buf_y = deque(maxlen=SMOOTH_WINDOW)
smoothed_x = screen_w // 2
smoothed_y = screen_h // 2

# pinch state (for scroll / double-click)
pinch_active = False
last_pinch_rise = 0.0
pinch_start_y = 0.0
pinch_start_time = 0.0
scroll_acc = 0.0

# stability
hand_frames = 0

# fist state
fist_active = False
fist_last_time = 0.0

# blink state
blink_last_time = 0.0
blink_status = False

# =========================
# Helpers
# =========================
def to_screen(x_norm, y_norm):
    """
    Map normalized MediaPipe coords to full screen.
    We scale around center so small hand motion maps to whole screen.
    Do NOT invert X here — the frame is flipped for display and already processed flipped.
    """
    # scale around center (0.5)
    cx = 0.5 + (x_norm - 0.5) * HORIZONTAL_SCALE
    cy = 0.5 + (y_norm - 0.5) * VERTICAL_SCALE

    # DO NOT mirror/invert here; frame was flipped before processing
    sx = int(max(0.0, min(1.0, cx)) * screen_w)
    sy = int(max(0.0, min(1.0, cy)) * screen_h)
    return sx, sy

def avg_deque(d):
    return sum(d) / len(d) if d else 0

# =========================
# Main loop (hand-only control + blink scroll)
# =========================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # mirror frame for display only so movements feel natural visually
        frame = cv2.flip(frame, 1) if MIRROR else frame.copy()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process both hands and face each frame
        hand_results = hands.process(img_rgb)
        face_results = face_mesh.process(img_rgb)

        hand_control_active = False
        now = time.time()
        blink_status = False

        # --- face / blink detection (scroll on blink) ---
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            # left eye landmarks (common indices)
            try:
                lx_left = face_landmarks[33].x
                lx_right = face_landmarks[133].x
                ly_top = face_landmarks[159].y
                ly_bottom = face_landmarks[145].y

                # eye aspect ratio (vertical / horizontal) — normalized
                ear = abs(ly_top - ly_bottom) / (abs(lx_right - lx_left) + 1e-9)

                if ear < BLINK_THRESHOLD and (now - blink_last_time) > BLINK_DEBOUNCE:
                    # blink detected -> scroll down
                    pyautogui.scroll(BLINK_SCROLL_AMOUNT)
                    blink_last_time = now
                    blink_status = True
            except Exception:
                pass

        # --- hand processing (existing logic) ---
        if hand_results.multi_hand_landmarks:
            hand_frames += 1
            hand = hand_results.multi_hand_landmarks[0]

            # landmarks
            ix = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            iy = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            ip_y = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            thumb_x = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_y = hand.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            # -----------------------
            # Fist detection (new)
            # -----------------------
            lm = hand.landmark
            folded_count = 0
            finger_pairs = [
                (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
                (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
                (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
                (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
            ]
            for tip, pip in finger_pairs:
                if lm[tip].y > lm[pip].y:
                    folded_count += 1

            fist_detected = (folded_count >= 4)

            # Rising edge: click once
            if fist_detected and (not fist_active) and (now - fist_last_time) > FIST_DEBOUNCE:
                pyautogui.click()
                fist_active = True
                fist_last_time = now
            elif not fist_detected:
                fist_active = False

            # normalized pinch distance (thumb <-> index)
            pinch_dist = math.hypot(thumb_x - ix, thumb_y - iy)

            # index-up heuristic: tip above PIP -> controlling
            index_up = iy < ip_y

            # --- Pinch detection (enter scroll mode while pinched) + double-pinch => doubleClick ---
            if not pinch_active and pinch_dist < PINCH_ENTER:
                # rising edge
                pinch_active = True
                pinch_start_y = iy
                pinch_start_time = now
                scroll_acc = 0.0
                # double pinch detection
                if 0 < (now - last_pinch_rise) <= DOUBLE_PINCH_TIME:
                    pyautogui.doubleClick()
                    last_pinch_rise = 0.0
                else:
                    last_pinch_rise = now

            elif pinch_active and pinch_dist > PINCH_EXIT:
                # falling edge
                pinch_active = False
                # no single-click on release (double-pinch used for selection)

            # --- While pinched: SCROLL mode (vertical movement) ---
            if pinch_active:
                # compute vertical movement relative to last baseline
                dy = pinch_start_y - iy  # positive if moved up
                if abs(dy) > SCROLL_DEADZONE:
                    scroll_delta = dy * SCROLL_SENSITIVITY
                    scroll_acc += scroll_delta
                    int_scroll = int(scroll_acc)
                    if int_scroll != 0:
                        pyautogui.scroll(int_scroll)
                        scroll_acc -= int_scroll
                        # update baseline for continuous scrolling
                        pinch_start_y = iy
                # while pinched we do not move cursor
                hand_control_active = False
            else:
                # --- Movement mode: move cursor with index (require stability) ---
                if index_up and hand_frames >= HAND_STABLE_FRAMES:
                    # NOTE: MediaPipe landmark x is relative to the flipped frame if MIRROR==True.
                    # Because we flipped the frame (display) and DID NOT invert coords, the mapping
                    # is now intuitive: moving right in front of camera moves cursor right.
                    sx, sy = to_screen(ix, iy)
                    pos_buf_x.append(sx)
                    pos_buf_y.append(sy)
                    avg_x = avg_deque(pos_buf_x)
                    avg_y = avg_deque(pos_buf_y)
                    smoothed_x = smoothed_x * (1 - EXP_SMOOTH) + avg_x * EXP_SMOOTH
                    smoothed_y = smoothed_y * (1 - EXP_SMOOTH) + avg_y * EXP_SMOOTH
                    pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
                    hand_control_active = True
                else:
                    pos_buf_x.clear()
                    pos_buf_y.clear()

        else:
            # no hand detected
            hand_frames = 0
            pos_buf_x.clear()
            pos_buf_y.clear()
            pinch_active = False

        # UI overlay
        mode_text = "Pinch=Scroll (hold), DoublePinch=DoubleClick, Fist=Click, Blink=ScrollDown"
        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        status = []
        if pinch_active:
            status.append("SCROLLING")
        if fist_active:
            status.append("FIST_CLICKED")
        if blink_status:
            status.append("BLINK_SCROLL")
        cv2.putText(frame, " | ".join(status), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        cv2.imshow("Hand-only Control (pinch-scroll, double-pinch dblclick, fist click, blink scroll)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    face_mesh.close()
