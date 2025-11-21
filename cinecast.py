"""
CineCast - Bimanual Cinematic Gesture Control System
====================================================
Left Hand:  Selects effect mode (zoom, rotate, blur, filter)
Right Hand: Controls effect intensity in real-time
Single Hand Mode: Left hand controls both mode (gesture) and intensity (height)
"""

import cv2
import numpy as np
import time
import mediapipe as mp
from collections import deque


# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_INDEX = 0  # iPhone via Continuity Camera

# Hand detection thresholds
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MAX_NUM_HANDS = 2

# Gesture detection
PINCH_DISTANCE_THRESHOLD = 0.025  # Stricter threshold (2.5% of frame width)
PINCH_DEPTH_THRESHOLD = 0.03      # Z-axis alignment requirement
PINCH_DEBOUNCE_TIME = 1.0         # seconds
GESTURE_STABILITY_FRAMES = 3      # Frames to confirm gesture change

# Effect smoothing parameters
INTENSITY_SMOOTHING = 0.15  # Lower = smoother (0-1)
MODE_TRANSITION_SPEED = 0.2  # Crossfade speed between effects

# Single-hand mode
SINGLE_HAND_MODE_TIMEOUT = 1.5  # Switch to single-hand after 1.5s with one hand

# Effect modes
MODE_NORMAL = 0
MODE_ZOOM = 1
MODE_ROTATE = 2
MODE_BLUR = 3
MODE_FILTER = 4

MODE_NAMES = {
    MODE_NORMAL: "Normal",
    MODE_ZOOM: "Dolly Zoom",
    MODE_ROTATE: "Rotate",
    MODE_BLUR: "Motion Blur",
    MODE_FILTER: "Color Grade"
}

# Gesture names for better UI
GESTURE_ICONS = {
    MODE_NORMAL: "✋",
    MODE_ZOOM: "👍",      # Thumb up
    MODE_ROTATE: "✌️",     # Peace sign
    MODE_BLUR: "🤘",      # Rock sign
    MODE_FILTER: "👌",    # OK sign
}


# ============================================================================
# GLOBAL STATE
# ============================================================================

# Video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

# Pause state
paused = False
paused_frame = None
last_pinch_toggle = 0

# Bimanual control state
current_mode = MODE_NORMAL
current_intensity = 0.5  # 0.0 to 1.0
smoothed_intensity = 0.5
previous_mode = MODE_NORMAL
mode_blend_factor = 1.0  # For smooth mode transitions

# Hand tracking state
left_hand_detected = False
right_hand_detected = False
last_two_hands_time = 0
single_hand_mode = False
gesture_buffer = deque(maxlen=GESTURE_STABILITY_FRAMES)  # For gesture debouncing

# Recording state
recording = False
video_writer = None
output_filename = None

# FPS tracking
fps_queue = deque(maxlen=30)
last_frame_time = time.time()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    max_num_hands=MAX_NUM_HANDS
)
mp_draw = mp.solutions.drawing_utils


# ============================================================================
# HAND DETECTION & GESTURE CLASSIFICATION
# ============================================================================

def is_pinch(landmarks, w, h):
    """
    Improved pinch detection with stricter requirements
    Requires thumb and index to be very close in 3D space
    """
    thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # 2D distance (in screen space)
    dx = (thumb.x - index.x) * w
    dy = (thumb.y - index.y) * h
    dist_2d = (dx**2 + dy**2) ** 0.5

    # Z-axis depth alignment
    dz = abs(thumb.z - index.z)

    # Additional check: middle finger should be far from pinch point
    middle_dist = ((middle.x - thumb.x)**2 + (middle.y - thumb.y)**2) ** 0.5

    # Stricter criteria
    is_pinching = (
        dist_2d < PINCH_DISTANCE_THRESHOLD * w and  # Very close in 2D
        dz < PINCH_DEPTH_THRESHOLD and              # Aligned in depth
        middle_dist > 0.05                           # Other fingers not involved
    )

    return is_pinching


def get_hand_openness(landmarks):
    """
    Calculate how open the hand is (0.0 = closed fist, 1.0 = wide open)
    Used for intensity control with right hand
    """
    # Calculate average distance of fingertips from palm center
    palm_center = landmarks[mp_hands.HandLandmark.WRIST]
    fingertips = [
        landmarks[mp_hands.HandLandmark.THUMB_TIP],
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.PINKY_TIP]
    ]

    total_dist = 0
    for tip in fingertips:
        dx = tip.x - palm_center.x
        dy = tip.y - palm_center.y
        dz = tip.z - palm_center.z
        dist = (dx**2 + dy**2 + dz**2) ** 0.5
        total_dist += dist

    avg_dist = total_dist / len(fingertips)

    # Normalize to 0-1 range (empirically determined bounds)
    min_dist = 0.1  # closed fist
    max_dist = 0.4  # wide open hand

    intensity = (avg_dist - min_dist) / (max_dist - min_dist)
    return np.clip(intensity, 0.0, 1.0)


def count_extended_fingers(landmarks):
    """
    Count how many fingers are extended
    Returns tuple: (count, [thumb, index, middle, ring, pinky])
    """
    fingers_extended = [False] * 5

    # Thumb: check if tip is far from palm in x-direction
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]

    # Thumb extended if tip is further from wrist than IP joint
    thumb_dist = abs(thumb_tip.x - wrist.x)
    thumb_ip_dist = abs(thumb_ip.x - wrist.x)
    fingers_extended[0] = thumb_dist > thumb_ip_dist * 1.2

    # Other fingers: tip above PIP joint
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
        fingers_extended[i + 1] = landmarks[tip].y < landmarks[pip].y

    return sum(fingers_extended), fingers_extended


def classify_mode_gesture(landmarks):
    """
    Improved gesture classification with distinct patterns

    Gestures (designed to be very different):
    - THUMB UP (only thumb extended): Zoom
    - PEACE SIGN (index + middle only): Rotate
    - ROCK SIGN (index + pinky only): Blur
    - OK SIGN (thumb + index touching, others up): Filter
    - OPEN PALM (all fingers): Normal
    """
    count, extended = count_extended_fingers(landmarks)

    # Check for OK sign first (thumb and index touching)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ok_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) ** 0.5

    if ok_dist < 0.05 and extended[2] and extended[3] and extended[4]:  # OK sign
        return MODE_FILTER

    # Thumb up (only thumb extended)
    if extended[0] and not extended[1] and not extended[2] and not extended[3] and not extended[4]:
        return MODE_ZOOM

    # Peace sign (index + middle only)
    if not extended[0] and extended[1] and extended[2] and not extended[3] and not extended[4]:
        return MODE_ROTATE

    # Rock sign (index + pinky only)
    if not extended[0] and extended[1] and not extended[2] and not extended[3] and extended[4]:
        return MODE_BLUR

    # Open palm (4-5 fingers extended) -> Normal
    if count >= 4:
        return MODE_NORMAL

    # Default to current mode if gesture unclear (prevents flickering)
    return None  # Will be handled by caller


def get_hand_height_normalized(landmarks):
    """
    Get vertical position of hand (0.0 = bottom, 1.0 = top)
    Used for single-hand intensity control
    """
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    # Invert Y (0 is top in image coords, we want 0 at bottom)
    height = 1.0 - wrist.y
    return np.clip(height, 0.0, 1.0)


def identify_left_right_hands(multi_hand_landmarks, multi_handedness):
    """
    Identify which detected hand is left vs right
    Returns: (left_hand_landmarks, right_hand_landmarks)
    """
    left_hand = None
    right_hand = None

    for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
        # MediaPipe returns handedness from camera perspective (mirrored)
        label = handedness.classification[0].label

        if label == "Left":  # Actually right hand (mirrored)
            right_hand = hand_landmarks
        else:  # Actually left hand (mirrored)
            left_hand = hand_landmarks

    return left_hand, right_hand


# ============================================================================
# EFFECT SYSTEM - Organized for easy expansion
# ============================================================================

def apply_zoom_effect(frame, intensity, t):
    """
    Dolly zoom effect
    intensity: 0.0 = no zoom, 1.0 = max zoom
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    # Map intensity to scale factor (0.7x to 1.5x zoom)
    scale = 0.7 + intensity * 0.8

    # Add subtle oscillation for cinematic feel
    scale += 0.05 * np.sin(t * 2.0) * intensity

    M = cv2.getRotationMatrix2D(center, 0, scale)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def apply_rotate_effect(frame, intensity, t):
    """
    Smooth rotation effect
    intensity: 0.0 = no rotation, 1.0 = fast rotation
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    # Rotation speed based on intensity
    angle = (t * 30 * intensity) % 360

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def apply_blur_effect(frame, intensity):
    """
    Variable motion blur / focus effect
    intensity: 0.0 = sharp, 1.0 = maximum blur
    """
    if intensity < 0.05:
        return frame

    # Map intensity to kernel size (must be odd)
    kernel_size = int(5 + intensity * 46)  # 5 to 51
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_size = max(5, min(51, kernel_size))

    # Use bilateral filter for edge-preserving blur (more cinematic)
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def apply_filter_effect(frame, intensity):
    """
    Color grading filter with intensity control
    intensity: 0.0 = original, 1.0 = full color grade
    """
    if intensity < 0.05:
        return frame

    # Convert to HSV for color manipulation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Cinematic color grade: warm shadows, cool highlights
    hsv[:, :, 0] = (hsv[:, :, 0] + intensity * 10) % 180  # Hue shift
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + intensity * 0.3), 0, 255)  # Saturation boost
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.9 + intensity * 0.2), 0, 255)  # Slight darkening

    hsv = hsv.astype(np.uint8)
    graded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Blend original with graded based on intensity
    return cv2.addWeighted(frame, 1 - intensity, graded, intensity, 0)


def apply_effect(frame, mode, intensity, t):
    """
    Main effect dispatcher with smooth blending
    """
    if mode == MODE_NORMAL:
        return frame
    elif mode == MODE_ZOOM:
        return apply_zoom_effect(frame, intensity, t)
    elif mode == MODE_ROTATE:
        return apply_rotate_effect(frame, intensity, t)
    elif mode == MODE_BLUR:
        return apply_blur_effect(frame, intensity)
    elif mode == MODE_FILTER:
        return apply_filter_effect(frame, intensity)

    return frame


# ============================================================================
# UI RENDERING
# ============================================================================

def draw_ui(frame, mode, intensity, fps, recording_status, left_detected, right_detected, single_mode):
    """
    Draw comprehensive UI overlay
    Shows: mode, intensity, hand status, FPS, recording indicator, gesture hints
    """
    h, w = frame.shape[:2]

    # Semi-transparent overlay panel (extended height for hand indicators)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Mode display with gesture icon
    mode_name = MODE_NAMES.get(mode, "Unknown")
    gesture_icon = GESTURE_ICONS.get(mode, "❓")
    cv2.putText(frame, f"{gesture_icon} MODE: {mode_name}", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (100, 255, 100), 2)

    # Intensity bar
    bar_x = 20
    bar_y = 60
    bar_width = 300
    bar_height = 20

    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (60, 60, 60), -1)

    # Intensity fill with gradient effect
    fill_width = int(bar_width * intensity)
    color_intensity = (
        int(100 + 155 * intensity),
        int(255 - 100 * intensity),
        100
    )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                  color_intensity, -1)

    # Intensity percentage
    cv2.putText(frame, f"Intensity: {int(intensity * 100)}%", (bar_x + bar_width + 20, bar_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hand detection status indicators
    hand_status_y = 95
    hand_status_x = 20

    # Control mode indicator
    if single_mode:
        cv2.putText(frame, "SINGLE HAND MODE", (hand_status_x, hand_status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        cv2.putText(frame, "(Gesture = Mode, Height = Intensity)", (hand_status_x, hand_status_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    else:
        # Left hand indicator
        left_color = (100, 255, 100) if left_detected else (80, 80, 80)
        left_status = "✓" if left_detected else "✗"
        cv2.putText(frame, f"L {left_status}", (hand_status_x, hand_status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)

        # Right hand indicator
        right_color = (100, 255, 100) if right_detected else (80, 80, 80)
        right_status = "✓" if right_detected else "✗"
        cv2.putText(frame, f"R {right_status}", (hand_status_x + 60, hand_status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)

        # Labels
        cv2.putText(frame, "L: Mode | R: Intensity", (hand_status_x + 120, hand_status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Recording indicator
    if recording_status:
        cv2.circle(frame, (w - 40, 60), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 100, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

    # Gesture guide in bottom-left corner
    draw_gesture_guide(frame, h, w)

    # Controls hint at bottom
    cv2.putText(frame, "Controls: [R] Record | [Q] Quit | [H] Toggle Hints",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


def draw_gesture_guide(frame, h, w):
    """Draw a small gesture reference guide"""
    guide_x = w - 250
    guide_y = h - 160
    guide_w = 230
    guide_h = 135

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Title
    cv2.putText(frame, "Gesture Guide:", (guide_x + 5, guide_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Gestures list
    gestures = [
        "👍 Thumbs Up = Zoom",
        "✌️  Peace = Rotate",
        "🤘 Rock = Blur",
        "👌 OK = Filter",
        "✋ Open = Normal"
    ]

    y_offset = guide_y + 45
    for gesture in gestures:
        cv2.putText(frame, gesture, (guide_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y_offset += 18


def draw_paused_overlay(frame):
    """Draw paused state overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark semi-transparent overlay
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    # PAUSED text
    text = "PAUSED"
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size = cv2.getTextSize(text, font, 2.5, 4)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2

    cv2.putText(frame, text, (text_x, text_y), font, 2.5, (0, 100, 255), 4)

    return frame


# ============================================================================
# VIDEO RECORDING
# ============================================================================

def start_recording(frame_width, frame_height):
    """Initialize video recording"""
    global video_writer, output_filename, recording

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"cinecast_output_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_filename, fourcc, 30.0,
                                   (frame_width, frame_height))
    recording = True
    print(f"[REC] Recording started: {output_filename}")


def stop_recording():
    """Stop and finalize video recording"""
    global video_writer, recording, output_filename

    if video_writer is not None:
        video_writer.release()
        video_writer = None
        recording = False
        print(f"[SAVE] Recording saved: {output_filename}")
        output_filename = None


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    global paused, paused_frame, last_pinch_toggle
    global current_mode, current_intensity, smoothed_intensity
    global previous_mode, mode_blend_factor
    global last_frame_time, recording
    global left_hand_detected, right_hand_detected, last_two_hands_time, single_hand_mode
    global gesture_buffer

    print("\n" + "="*65)
    print("  CINECAST - Bimanual Cinematic Gesture Control")
    print("="*65)
    print("\nGESTURES (Much More Distinct!):")
    print("  👍 THUMBS UP:     Dolly Zoom")
    print("  ✌️  PEACE SIGN:    Rotate")
    print("  🤘 ROCK SIGN:     Motion Blur")
    print("  👌 OK SIGN:       Color Grade")
    print("  ✋ OPEN PALM:     Normal/Reset")
    print("\nTWO-HAND MODE:")
    print("  Left Hand:  Gesture selects mode")
    print("  Right Hand: Openness controls intensity (0-100%)")
    print("\nSINGLE-HAND MODE (auto-activates after 1.5s):")
    print("  Gesture:    Selects mode")
    print("  Height:     Controls intensity (raise = more intense)")
    print("\nCONTROLS:")
    print("  [R] Start/Stop Recording")
    print("  [Q] Quit")
    print("  Pinch: Pause/Resume (stricter detection)")
    print("="*65 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        # FPS calculation
        current_time = time.time()
        fps_queue.append(1.0 / (current_time - last_frame_time))
        last_frame_time = current_time
        avg_fps = sum(fps_queue) / len(fps_queue)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand detection
        results = hands.process(rgb)

        # Update hand detection state
        left_hand_detected = False
        right_hand_detected = False

        # Process hands for bimanual control
        if results.multi_hand_landmarks and results.multi_handedness:
            left_hand, right_hand = identify_left_right_hands(
                results.multi_hand_landmarks,
                results.multi_handedness
            )

            left_hand_detected = left_hand is not None
            right_hand_detected = right_hand is not None

            # Track when we last had two hands
            if left_hand_detected and right_hand_detected:
                last_two_hands_time = current_time
                single_hand_mode = False
            # Switch to single-hand mode after timeout
            elif (left_hand_detected or right_hand_detected):
                if current_time - last_two_hands_time > SINGLE_HAND_MODE_TIMEOUT:
                    if not single_hand_mode:
                        single_hand_mode = True
                        print("[MODE] Switched to SINGLE-HAND mode")

            # Draw hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
                )

            # SINGLE-HAND MODE: Use one hand for both gesture and intensity
            if single_hand_mode:
                active_hand = left_hand if left_hand is not None else right_hand

                if active_hand is not None:
                    # Check for pinch first
                    if is_pinch(active_hand.landmark, w, h):
                        now = time.time()
                        if now - last_pinch_toggle > PINCH_DEBOUNCE_TIME:
                            paused = not paused
                            if paused:
                                paused_frame = frame.copy()
                            last_pinch_toggle = now
                            print(f"[PAUSE] {'ON' if paused else 'OFF'}")
                    else:
                        # Gesture determines mode
                        detected_gesture = classify_mode_gesture(active_hand.landmark)

                        if detected_gesture is not None:
                            gesture_buffer.append(detected_gesture)

                            # Only change mode if gesture is stable
                            if len(gesture_buffer) == GESTURE_STABILITY_FRAMES:
                                if all(g == detected_gesture for g in gesture_buffer):
                                    if detected_gesture != current_mode:
                                        previous_mode = current_mode
                                        current_mode = detected_gesture
                                        mode_blend_factor = 0.0
                                        print(f"[MODE] {MODE_NAMES[current_mode]}")

                        # Hand height determines intensity
                        height_intensity = get_hand_height_normalized(active_hand.landmark)
                        current_intensity = height_intensity

            # TWO-HAND MODE: Left = gesture, Right = intensity
            else:
                # Left hand: Mode selection
                if left_hand is not None:
                    # Check for pinch to pause
                    if is_pinch(left_hand.landmark, w, h):
                        now = time.time()
                        if now - last_pinch_toggle > PINCH_DEBOUNCE_TIME:
                            paused = not paused
                            if paused:
                                paused_frame = frame.copy()
                            last_pinch_toggle = now
                            print(f"[PAUSE] {'ON' if paused else 'OFF'}")
                    else:
                        detected_gesture = classify_mode_gesture(left_hand.landmark)

                        if detected_gesture is not None:
                            gesture_buffer.append(detected_gesture)

                            # Only change mode if gesture is stable
                            if len(gesture_buffer) == GESTURE_STABILITY_FRAMES:
                                if all(g == detected_gesture for g in gesture_buffer):
                                    if detected_gesture != current_mode:
                                        previous_mode = current_mode
                                        current_mode = detected_gesture
                                        mode_blend_factor = 0.0
                                        print(f"[MODE] {MODE_NAMES[current_mode]}")

                # Right hand: Intensity control
                if right_hand is not None:
                    # Check for pinch to pause
                    if is_pinch(right_hand.landmark, w, h):
                        now = time.time()
                        if now - last_pinch_toggle > PINCH_DEBOUNCE_TIME:
                            paused = not paused
                            if paused:
                                paused_frame = frame.copy()
                            last_pinch_toggle = now
                            print(f"[PAUSE] {'ON' if paused else 'OFF'}")
                    else:
                        raw_intensity = get_hand_openness(right_hand.landmark)
                        current_intensity = raw_intensity
        else:
            # No hands detected - reset to two-hand mode after delay
            if single_hand_mode and current_time - last_two_hands_time > 3.0:
                single_hand_mode = False
                gesture_buffer.clear()

        # Smooth intensity transitions
        smoothed_intensity += (current_intensity - smoothed_intensity) * INTENSITY_SMOOTHING

        # Smooth mode transitions
        if mode_blend_factor < 1.0:
            mode_blend_factor = min(1.0, mode_blend_factor + MODE_TRANSITION_SPEED)

        # Handle paused state
        if paused and paused_frame is not None:
            output = draw_paused_overlay(paused_frame.copy())
        else:
            # Apply current effect
            t = time.time()
            output = apply_effect(frame, current_mode, smoothed_intensity, t)

            # Blend with previous mode for smooth transitions
            if mode_blend_factor < 1.0 and previous_mode != current_mode:
                previous_output = apply_effect(frame, previous_mode, smoothed_intensity, t)
                output = cv2.addWeighted(previous_output, 1 - mode_blend_factor,
                                        output, mode_blend_factor, 0)

        # Draw UI overlay
        output = draw_ui(output, current_mode, smoothed_intensity, avg_fps, recording,
                        left_hand_detected, right_hand_detected, single_hand_mode)

        # Write frame if recording
        if recording and video_writer is not None:
            video_writer.write(output)

        # Display
        cv2.imshow("CineCast - Bimanual Control", output)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[EXIT] Shutting down...")
            break
        elif key == ord('r') or key == ord('R'):
            if recording:
                stop_recording()
            else:
                start_recording(w, h)

    # Cleanup
    if recording:
        stop_recording()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[EXIT] Cleanup complete. Goodbye!\n")


if __name__ == "__main__":
    main()
