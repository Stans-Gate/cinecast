"""
CineCast - Single-Hand Cinematic Gesture Control System
========================================================
Main application file - orchestrates all modules

One hand controls everything:
1. Start in NO MODE (passthrough)
2. Menu visible when unlocked - scroll and select modes
3. Control intensity with palm open/close (when locked)
4. Fist gesture → QUIT back to NO MODE
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
    classify_mode_gesture,
    detect_menu_scroll_gesture,
    detect_menu_select_gesture,
    get_hand_position,
    detect_3d_rotation_gesture,
    detect_3d_scale_gesture
)
from ui_renderer import draw_ui, draw_gesture_guide, draw_mode_menu


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
QUIT_GESTURE_FRAMES = 5  # Reduced for faster quit detection
MENU_SCROLL_STABILITY_FRAMES = 8  # Increased for slower, more controlled scrolling
MENU_SELECT_STABILITY_FRAMES = 6  # Reduced for faster OK sign selection

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

# Menu state
menu_visible = True  # Menu visible when unlocked
selected_menu_index = 0  # Currently selected menu item
menu_scroll_buffer = deque(maxlen=MENU_SCROLL_STABILITY_FRAMES)
menu_select_buffer = deque(maxlen=MENU_SELECT_STABILITY_FRAMES)
last_scroll_direction = None  # Track last scroll direction for hysteresis
SCROLL_COOLDOWN = 0.8  # Minimum seconds between scrolls (increased for slower scrolling)

# Gesture tracking
hand_detected = False
gesture_buffer = deque(maxlen=GESTURE_STABILITY_FRAMES)
quit_buffer = deque(maxlen=QUIT_GESTURE_FRAMES)
previous_hand_position = None  # For 3D rotation tracking

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
    global menu_visible, selected_menu_index, menu_scroll_buffer, menu_select_buffer
    global previous_hand_position, last_scroll_direction
    
    # Initialize scroll timing (local variable that persists across iterations)
    last_scroll_time_local = 0.0

    print("\n" + "="*70)
    print("  CINECAST - Single-Hand Cinematic Gesture Control")
    print("="*70)
    print("\n🎬 HOW IT WORKS:")
    print("  1. Start in NO MODE (camera passthrough)")
    print("  2. Menu visible when unlocked - scroll and select modes")
    print("  3. Control intensity by opening/closing palm (when locked)")
    print("  4. Fist gesture → QUIT back to NO MODE")
    print("\n🔓 AVAILABLE EFFECTS:")
    for i, effect in enumerate(AVAILABLE_EFFECTS):
        print(f"  {i+1}. {effect.icon} {effect.name}")
    print("\n📜 MENU CONTROLS (when unlocked):")
    print("  • Index finger pointing up → Scroll up through menu")
    print("  • Index finger pointing down → Scroll down through menu")
    print("  • OK sign (thumb + index) → Select mode")
    print("\n🔒 WHEN LOCKED:")
    print("  • Fist closed = 0% intensity")
    print("  • Palm open = 100% intensity")
    print("  • 3D Object mode: Move hand to rotate, palm openness to scale")
    print("\n👎 QUIT GESTURE:")
    print("  • Thumbs down → unlocks and returns to NO MODE")
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
        
        # Store results for effect application (needed for Iron Man effect)
        current_hand_landmarks = None

        # SINGLE-HAND LOCK/UNLOCK SYSTEM
        if results.multi_hand_landmarks:
            active_hand = results.multi_hand_landmarks[0]
            hand_detected = True
            landmarks = active_hand.landmark
            current_hand_landmarks = landmarks  # Store for effects that need it

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, active_hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

            # Get current hand position for tracking
            current_hand_position = get_hand_position(landmarks)

            # Check for QUIT gesture (fist)
            if is_quit_gesture(landmarks):
                quit_buffer.append(True)

                if len(quit_buffer) == QUIT_GESTURE_FRAMES:
                    if all(quit_buffer):
                        if mode_locked:
                            mode_locked = False
                            menu_visible = True
                            if current_effect:
                                current_effect.reset()
                            current_effect = None
                            gesture_buffer.clear()
                            menu_scroll_buffer.clear()
                            menu_select_buffer.clear()
                            previous_hand_position = None
                            last_scroll_direction = None
                            print("[UNLOCK] Returned to NO MODE")
            else:
                quit_buffer.append(False)

            # MODE LOCKED: Control intensity and mode-specific interactions
            if mode_locked:
                raw_intensity = get_palm_openness(landmarks)
                current_intensity = raw_intensity
                
                # Special handling for 3D Object mode
                if current_effect and current_effect.mode_id == 5:  # Object3DEffect
                    # Update rotation based on hand orientation (direct mapping)
                    # Get previous hand angles from effect
                    prev_angles = (current_effect.previous_hand_angle_x, current_effect.previous_hand_angle_y)
                    hand_angle_x, hand_angle_y = detect_3d_rotation_gesture(landmarks, prev_angles)
                    
                    # Only update if we have valid angles (hand detected)
                    if hand_angle_x is not None and hand_angle_y is not None:
                        current_effect.update_rotation(hand_angle_x, hand_angle_y)
                        # Store current angles for next frame
                        current_effect.previous_hand_angle_x = hand_angle_x
                        current_effect.previous_hand_angle_y = hand_angle_y
                    
                    # Update scale based on palm openness
                    scale = detect_3d_scale_gesture(landmarks)
                    current_effect.update_scale(scale)
                
                # Update tracking for next frame
                previous_hand_position = current_hand_position

            # MODE UNLOCKED: Menu navigation and selection
            else:
                menu_visible = True
                
                # Menu scrolling using index finger up/down gestures with cooldown
                scroll_direction = detect_menu_scroll_gesture(landmarks)
                if scroll_direction:
                    menu_scroll_buffer.append(scroll_direction)
                    if len(menu_scroll_buffer) >= MENU_SCROLL_STABILITY_FRAMES:
                        # Check if most recent scrolls are in the same direction
                        recent_directions = list(menu_scroll_buffer)[-MENU_SCROLL_STABILITY_FRAMES:]
                        same_direction_count = sum(1 for d in recent_directions if d == scroll_direction)
                        
                        # Require at least 2/3 of frames to be same direction for stability
                        # Also check cooldown to prevent scrolling too fast
                        time_since_last_scroll = current_time - last_scroll_time_local
                        if same_direction_count >= (MENU_SCROLL_STABILITY_FRAMES * 2 // 3) and time_since_last_scroll >= SCROLL_COOLDOWN:
                            if scroll_direction == 'up' and selected_menu_index > 0:
                                selected_menu_index -= 1
                                last_scroll_direction = 'up'
                                last_scroll_time_local = current_time
                                menu_scroll_buffer.clear()
                            elif scroll_direction == 'down' and selected_menu_index < len(AVAILABLE_EFFECTS) - 1:
                                selected_menu_index += 1
                                last_scroll_direction = 'down'
                                last_scroll_time_local = current_time
                                menu_scroll_buffer.clear()
                else:
                    # Reset scroll buffer if gesture not detected
                    menu_scroll_buffer.clear()
                
                # Menu selection (OK sign - thumb and index touching)
                if detect_menu_select_gesture(landmarks):
                    menu_select_buffer.append(True)
                    if len(menu_select_buffer) == MENU_SELECT_STABILITY_FRAMES:
                        if all(menu_select_buffer):
                            # Lock into selected mode (ensure index is valid)
                            if 0 <= selected_menu_index < len(AVAILABLE_EFFECTS):
                                mode_locked = True
                                menu_visible = False
                                previous_effect = current_effect
                                selected_effect = AVAILABLE_EFFECTS[selected_menu_index]
                                current_effect = selected_effect
                                mode_blend_factor = 0.0
                                gesture_buffer.clear()
                                quit_buffer.clear()
                                menu_scroll_buffer.clear()
                                menu_select_buffer.clear()
                                previous_hand_position = get_hand_position(landmarks)  # Initialize for 3D tracking
                                last_scroll_direction = None
                                print(f"[LOCK] Mode locked: {current_effect.name}")
                            else:
                                # Reset invalid index
                                selected_menu_index = max(0, min(selected_menu_index, len(AVAILABLE_EFFECTS) - 1))
                                menu_select_buffer.clear()
                else:
                    menu_select_buffer.append(False)
                    if len(menu_select_buffer) > 0 and not menu_select_buffer[-1]:
                        # Clear buffer if gesture breaks
                        if menu_select_buffer.count(False) > 2:
                            menu_select_buffer.clear()
                
        else:
            hand_detected = False
            gesture_buffer.clear()
            quit_buffer.clear()
            menu_scroll_buffer.clear()
            menu_select_buffer.clear()
            previous_hand_position = None
            last_scroll_direction = None

        # Smooth intensity transitions
        smoothed_intensity += (current_intensity - smoothed_intensity) * INTENSITY_SMOOTHING

        # Smooth mode transitions
        if mode_blend_factor < 1.0:
            mode_blend_factor = min(1.0, mode_blend_factor + MODE_TRANSITION_SPEED)

        # Apply current effect
        t = time.time()
        if current_effect:
            # Special handling for Iron Man effect - needs hand landmarks
            if current_effect.mode_id == 6 and hand_detected and current_hand_landmarks is not None:
                current_effect.set_hand_landmarks(current_hand_landmarks)
            
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
        output = draw_mode_menu(output, AVAILABLE_EFFECTS, selected_menu_index, menu_visible and not mode_locked)
        output = draw_gesture_guide(output, mode_locked, AVAILABLE_EFFECTS, current_effect)

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
