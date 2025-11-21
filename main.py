"""
CineCast - Single-Hand Cinematic Gesture Control System
========================================================
Main application file - orchestrates all modules

One hand controls everything:
1. Start in NO MODE (passthrough)
2. Make a gesture → LOCKS into that mode
3. Control intensity with palm open/close
4. Middle finger up → QUIT back to NO MODE
"""

import cv2
import time
import mediapipe as mp
from collections import deque

# Import our modules
from effects import AVAILABLE_EFFECTS
from gesture_recognition import (
    get_palm_openness,
    is_quit_gesture,
    classify_mode_gesture
)
from ui_renderer import draw_ui, draw_gesture_guide


# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_INDEX = 0

# Hand detection
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MAX_NUM_HANDS = 2

# Gesture detection
GESTURE_STABILITY_FRAMES = 8
QUIT_GESTURE_FRAMES = 10

# Effect smoothing
INTENSITY_SMOOTHING = 0.15
MODE_TRANSITION_SPEED = 0.2


# ============================================================================
# INITIALIZATION
# ============================================================================

# Video capture
cap = cv2.VideoCapture(CAMERA_INDEX)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    max_num_hands=MAX_NUM_HANDS
)
mp_draw = mp.solutions.drawing_utils

# Build effect lookup by mode_id
effects_by_id = {effect.mode_id: effect for effect in AVAILABLE_EFFECTS}

# Application state
mode_locked = False
current_effect = None  # Currently active effect
current_intensity = 0.5
smoothed_intensity = 0.5
previous_effect = None
mode_blend_factor = 1.0

# Gesture tracking
hand_detected = False
gesture_buffer = deque(maxlen=GESTURE_STABILITY_FRAMES)
quit_buffer = deque(maxlen=QUIT_GESTURE_FRAMES)

# Recording
recording = False
video_writer = None
output_filename = None

# FPS tracking
fps_queue = deque(maxlen=30)
last_frame_time = time.time()


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
    global current_effect, current_intensity, smoothed_intensity
    global previous_effect, mode_blend_factor
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
    print("\n🔓 AVAILABLE EFFECTS:")
    for effect in AVAILABLE_EFFECTS:
        print(f"  {effect.icon} {effect.name}")
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
        hand_detected = False

        # SINGLE-HAND LOCK/UNLOCK SYSTEM
        if results.multi_hand_landmarks:
            active_hand = results.multi_hand_landmarks[0]
            hand_detected = True

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, active_hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Check for QUIT gesture
            if is_quit_gesture(active_hand.landmark):
                quit_buffer.append(True)

                if len(quit_buffer) == QUIT_GESTURE_FRAMES:
                    if all(quit_buffer):
                        if mode_locked:
                            mode_locked = False
                            if current_effect:
                                current_effect.reset()
                            current_effect = None
                            gesture_buffer.clear()
                            print("[UNLOCK] Returned to NO MODE")
            else:
                quit_buffer.append(False)

            # MODE LOCKED: Only control intensity
            if mode_locked:
                raw_intensity = get_palm_openness(active_hand.landmark)
                current_intensity = raw_intensity

            # MODE UNLOCKED: Look for gesture to lock a mode
            else:
                detected_mode_id = classify_mode_gesture(active_hand.landmark, AVAILABLE_EFFECTS)

                if detected_mode_id is not None:
                    gesture_buffer.append(detected_mode_id)

                    if len(gesture_buffer) == GESTURE_STABILITY_FRAMES:
                        if all(g == detected_mode_id for g in gesture_buffer):
                            # Lock into this mode
                            mode_locked = True
                            previous_effect = current_effect
                            current_effect = effects_by_id.get(detected_mode_id)
                            mode_blend_factor = 0.0
                            gesture_buffer.clear()
                            quit_buffer.clear()
                            print(f"[LOCK] Mode locked: {current_effect.name}")
                else:
                    gesture_buffer.clear()
        else:
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
        if current_effect:
            output = current_effect.apply(frame, smoothed_intensity, t)

            # Blend with previous effect for smooth transitions
            if mode_blend_factor < 1.0 and previous_effect and previous_effect != current_effect:
                previous_output = previous_effect.apply(frame, smoothed_intensity, t)
                output = cv2.addWeighted(previous_output, 1 - mode_blend_factor,
                                        output, mode_blend_factor, 0)
        else:
            output = frame  # Passthrough (no effect)

        # Get current effect info for UI
        effect_name = current_effect.name if current_effect else "No Mode (Passthrough)"
        effect_icon = current_effect.icon if current_effect else "🎬"

        # Draw UI overlay
        output = draw_ui(output, effect_name, effect_icon, smoothed_intensity, avg_fps,
                        recording, hand_detected, mode_locked)
        output = draw_gesture_guide(output, mode_locked, AVAILABLE_EFFECTS)

        # Write frame if recording
        if recording and video_writer is not None:
            video_writer.write(output)

        # Display
        cv2.imshow("CineCast - Single-Hand Control", output)

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
