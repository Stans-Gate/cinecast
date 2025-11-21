"""
CineCast - Single-Hand Cinematic Gesture Control System
========================================================
One hand controls everything:
1. Start in NO MODE (passthrough)
2. Make a gesture → LOCKS into that mode
3. Control intensity with palm open/close
4. Middle finger up → QUIT back to NO MODE
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
GESTURE_STABILITY_FRAMES = 8      # Frames to confirm gesture (hold for ~0.5s at 15fps)
QUIT_GESTURE_FRAMES = 10          # Frames to confirm quit gesture (more stable)

# Effect smoothing parameters
INTENSITY_SMOOTHING = 0.15  # Lower = smoother (0-1)
MODE_TRANSITION_SPEED = 0.2  # Crossfade speed between effects

# Effect modes
MODE_NONE = -1        # No mode active (passthrough)
MODE_ZOOM = 1
MODE_ROTATE = 2
MODE_BLUR = 3
MODE_FILTER = 4

MODE_NAMES = {
    MODE_NONE: "No Mode (Passthrough)",
    MODE_ZOOM: "Dolly Zoom",
    MODE_ROTATE: "Rotate",
    MODE_BLUR: "Motion Blur",
    MODE_FILTER: "Color Grade"
}

# Gesture names for better UI
GESTURE_ICONS = {
    MODE_NONE: "🎬",
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

# Single-hand control state
mode_locked = False       # Is a mode currently locked?
current_mode = MODE_NONE  # Current active mode
current_intensity = 0.5   # 0.0 to 1.0
smoothed_intensity = 0.5
previous_mode = MODE_NONE
mode_blend_factor = 1.0   # For smooth mode transitions

# Gesture tracking
hand_detected = False
gesture_buffer = deque(maxlen=GESTURE_STABILITY_FRAMES)  # For mode lock gestures
quit_buffer = deque(maxlen=QUIT_GESTURE_FRAMES)           # For quit gesture

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

def get_palm_openness(landmarks):
    """
    Calculate palm openness based on finger curl
    Closed fist (all fingers curled) = 0.0
    Open palm (all fingers extended) = 1.0
    """
    # Count how many fingers are extended
    count, extended = count_extended_fingers(landmarks)

    # Calculate openness as ratio of extended fingers
    # 0 fingers = 0%, 5 fingers = 100%
    openness = count / 5.0

    # Add some granularity by checking finger curl angles
    finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky
    finger_mcp = [5, 9, 13, 17]    # knuckles

    curl_distances = []
    for tip, mcp in zip(finger_tips, finger_mcp):
        tip_pos = landmarks[tip]
        mcp_pos = landmarks[mcp]
        # Distance from tip to knuckle (smaller = more curled)
        dist = ((tip_pos.x - mcp_pos.x)**2 + (tip_pos.y - mcp_pos.y)**2) ** 0.5
        curl_distances.append(dist)

    avg_curl = np.mean(curl_distances)

    # Normalize curl distance (empirical values)
    # Closed fist: ~0.05, Open palm: ~0.15
    curl_intensity = (avg_curl - 0.05) / (0.15 - 0.05)
    curl_intensity = np.clip(curl_intensity, 0.0, 1.0)

    # Blend both metrics (60% curl, 40% finger count)
    final_openness = 0.6 * curl_intensity + 0.4 * openness

    return np.clip(final_openness, 0.0, 1.0)


def count_extended_fingers(landmarks):
    """
    Count how many fingers are extended (more robust detection)
    Returns tuple: (count, [thumb, index, middle, ring, pinky])
    """
    fingers_extended = [False] * 5

    # Thumb: check if tip is far from palm in x-direction
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]

    # Thumb extended if tip is further from wrist than IP joint
    thumb_tip_dist = abs(thumb_tip.x - wrist.x)
    thumb_ip_dist = abs(thumb_ip.x - wrist.x)
    fingers_extended[0] = thumb_tip_dist > thumb_ip_dist * 1.3

    # Other fingers: tip above PIP joint with better threshold
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    finger_mcps = [5, 9, 13, 17]

    for i, (tip, pip, mcp) in enumerate(zip(finger_tips, finger_pips, finger_mcps)):
        # Check if tip is significantly above the PIP joint
        tip_y = landmarks[tip].y
        pip_y = landmarks[pip].y
        mcp_y = landmarks[mcp].y

        # Finger is extended if tip is above PIP and the angle looks right
        is_extended = (tip_y < pip_y - 0.02) and (tip_y < mcp_y)
        fingers_extended[i + 1] = is_extended

    return sum(fingers_extended), fingers_extended


def is_quit_gesture(landmarks):
    """
    Detect QUIT gesture: Middle finger up only (very distinct)
    This should be hard to trigger accidentally
    """
    count, extended = count_extended_fingers(landmarks)

    # Middle finger up ONLY (index 2)
    # All other fingers must be down
    if extended[2] and not extended[0] and not extended[1] and not extended[3] and not extended[4]:
        if count == 1:
            return True

    return False


def classify_mode_gesture(landmarks):
    """
    Classify gesture to lock into a mode
    Only used when NOT in a locked mode

    Gestures (designed to be very different):
    - THUMB UP (only thumb extended): Zoom
    - PEACE SIGN (index + middle): Rotate
    - ROCK SIGN (index + pinky): Blur
    - OK SIGN (thumb + index touching, others up): Filter
    """
    count, extended = count_extended_fingers(landmarks)

    # Get thumb and index positions for OK sign check
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ok_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) ** 0.5

    # OK SIGN: thumb touches index, other 3 fingers up (relaxed requirement)
    if ok_dist < 0.06 and (extended[2] or extended[3] or extended[4]):
        # At least 2 of the 3 other fingers should be up
        others_up = sum([extended[2], extended[3], extended[4]])
        if others_up >= 2:
            return MODE_FILTER

    # THUMBS UP: only thumb extended, all others clearly down
    if extended[0] and count == 1:
        return MODE_ZOOM

    # PEACE SIGN: index + middle up, thumb/ring/pinky down
    if extended[1] and extended[2] and not extended[3] and not extended[4]:
        if count == 2 or (count == 3 and extended[0]):  # Allow thumb to be up too
            return MODE_ROTATE

    # ROCK SIGN: index + pinky up, middle/ring down
    if extended[1] and extended[4] and not extended[2] and not extended[3]:
        if count == 2 or (count == 3 and extended[0]):  # Allow thumb to be up too
            return MODE_BLUR

    # Ambiguous gesture - return None
    return None




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
    if mode == MODE_NONE:
        return frame  # Passthrough
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

def draw_ui(frame, mode, intensity, fps, recording_status, hand_detected, locked):
    """
    Draw comprehensive UI overlay
    Shows: mode, lock status, intensity, hand detection, FPS, recording
    """
    h, w = frame.shape[:2]

    # Semi-transparent overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Mode display with lock status
    mode_name = MODE_NAMES.get(mode, "Unknown")
    gesture_icon = GESTURE_ICONS.get(mode, "❓")

    if locked:
        lock_icon = "🔒"
        mode_color = (100, 255, 255)  # Cyan when locked
        status_text = "LOCKED"
    else:
        lock_icon = "🔓"
        mode_color = (100, 255, 100)  # Green when unlocked
        status_text = "UNLOCKED"

    cv2.putText(frame, f"{lock_icon} {gesture_icon} {mode_name}", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, mode_color, 2)

    # Status indicator
    cv2.putText(frame, status_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    # Intensity bar (only show when locked in a mode)
    if locked and mode != MODE_NONE:
        bar_x = 20
        bar_y = 90
        bar_width = 300
        bar_height = 20

        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (60, 60, 60), -1)

        # Intensity fill
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

    # Hand detection indicator
    hand_status_y = 125
    hand_color = (100, 255, 100) if hand_detected else (80, 80, 80)
    hand_icon = "👋" if hand_detected else "🚫"
    cv2.putText(frame, f"{hand_icon} Hand: {'Detected' if hand_detected else 'Not Detected'}",
                (20, hand_status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Recording indicator
    if recording_status:
        cv2.circle(frame, (w - 40, 60), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 100, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

    # Gesture guide in bottom-right corner
    draw_gesture_guide(frame, h, w, locked)

    # Controls hint at bottom
    cv2.putText(frame, "Controls: [R] Record | [Q] Quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


def draw_gesture_guide(frame, h, w, locked):
    """Draw a small gesture reference guide"""
    guide_x = w - 280
    guide_y = h - 185
    guide_w = 260
    guide_h = 165

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Title
    cv2.putText(frame, "Gesture Guide:", (guide_x + 5, guide_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if locked:
        # Show quit instruction when locked
        instructions = [
            "🔒 MODE LOCKED",
            "",
            "Close/Open palm = Intensity",
            "",
            "🖕 Middle Finger Up = QUIT"
        ]
    else:
        # Show available modes when unlocked
        instructions = [
            "Make gesture to LOCK mode:",
            "👍 Thumbs Up = Zoom",
            "✌️  Peace = Rotate",
            "🤘 Rock = Blur",
            "👌 OK = Filter"
        ]

    y_offset = guide_y + 45
    for instruction in instructions:
        cv2.putText(frame, instruction, (guide_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y_offset += 22


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
    global current_mode, current_intensity, smoothed_intensity
    global previous_mode, mode_blend_factor
    global last_frame_time, recording
    global hand_detected, mode_locked
    global gesture_buffer, quit_buffer

    print("\n" + "="*70)
    print("  CINECAST - Single-Hand Cinematic Gesture Control")
    print("="*70)
    print("\n🎬 HOW IT WORKS:")
    print("  1. Start in NO MODE (camera passthrough)")
    print("  2. Make a gesture → LOCKS into that mode")
    print("  3. Control intensity by opening/closing palm")
    print("  4. Middle finger up → QUIT to NO MODE")
    print("\n🔓 GESTURES TO LOCK A MODE (hold for 0.5s):")
    print("  👍 THUMBS UP:        Dolly Zoom")
    print("  ✌️  PEACE SIGN:       Rotate")
    print("  🤘 ROCK SIGN:        Motion Blur")
    print("  👌 OK SIGN:          Color Grade")
    print("\n🔒 WHEN LOCKED:")
    print("  • Fist closed = 0% intensity")
    print("  • Palm open = 100% intensity")
    print("  • Other gestures are IGNORED (no accidental switching!)")
    print("\n🖕 QUIT GESTURE:")
    print("  • Middle finger up ONLY → unlocks and returns to NO MODE")
    print("\n⌨️  KEYBOARD CONTROLS:")
    print("  [R] Start/Stop Recording")
    print("  [Q] Quit Application")
    print("="*70 + "\n")

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
        hand_detected = False

        # SINGLE-HAND LOCK/UNLOCK SYSTEM
        if results.multi_hand_landmarks:
            # Use first detected hand (supports any hand)
            active_hand = results.multi_hand_landmarks[0]
            hand_detected = True

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, active_hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Check for QUIT gesture (middle finger up)
            if is_quit_gesture(active_hand.landmark):
                quit_buffer.append(True)

                # If quit gesture is stable, unlock mode
                if len(quit_buffer) == QUIT_GESTURE_FRAMES:
                    if all(quit_buffer):  # All frames must be quit gesture
                        if mode_locked:
                            mode_locked = False
                            current_mode = MODE_NONE
                            gesture_buffer.clear()
                            print("[UNLOCK] Returned to NO MODE")
            else:
                quit_buffer.append(False)

            # MODE LOCKED: Only control intensity
            if mode_locked:
                # Intensity control with palm openness
                raw_intensity = get_palm_openness(active_hand.landmark)
                current_intensity = raw_intensity

            # MODE UNLOCKED: Look for gesture to lock a mode
            else:
                detected_gesture = classify_mode_gesture(active_hand.landmark)

                if detected_gesture is not None:
                    gesture_buffer.append(detected_gesture)

                    # Only lock mode if gesture is stable
                    if len(gesture_buffer) == GESTURE_STABILITY_FRAMES:
                        if all(g == detected_gesture for g in gesture_buffer):
                            # Lock into this mode
                            mode_locked = True
                            previous_mode = current_mode
                            current_mode = detected_gesture
                            mode_blend_factor = 0.0
                            gesture_buffer.clear()
                            quit_buffer.clear()
                            print(f"[LOCK] Mode locked: {MODE_NAMES[current_mode]}")
                else:
                    # Clear buffer if gesture is ambiguous
                    gesture_buffer.clear()
        else:
            # No hand detected
            hand_detected = False
            gesture_buffer.clear()
            quit_buffer.clear()

        # Smooth intensity transitions
        smoothed_intensity += (current_intensity - smoothed_intensity) * INTENSITY_SMOOTHING

        # Smooth mode transitions
        if mode_blend_factor < 1.0:
            mode_blend_factor = min(1.0, mode_blend_factor + MODE_TRANSITION_SPEED)

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
                        hand_detected, mode_locked)

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
