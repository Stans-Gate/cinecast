"""
UI Rendering Module
===================
All UI drawing functions
"""

import cv2


def draw_ui(frame, effect_name, effect_icon, intensity, fps, recording_status, hand_detected, locked):
    """
    Draw comprehensive UI overlay
    """
    h, w = frame.shape[:2]

    # Semi-transparent overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Lock status
    if locked:
        lock_icon = "🔒"
        mode_color = (100, 255, 255)  # Cyan when locked
        status_text = "LOCKED"
    else:
        lock_icon = "🔓"
        mode_color = (100, 255, 100)  # Green when unlocked
        status_text = "UNLOCKED"

    # Mode display
    cv2.putText(frame, f"{lock_icon} {effect_icon} {effect_name}", (20, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, mode_color, 2)

    cv2.putText(frame, status_text, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

    # Intensity bar (only when locked)
    if locked and effect_name != "No Mode (Passthrough)":
        bar_x = 20
        bar_y = 90
        bar_width = 300
        bar_height = 20

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (60, 60, 60), -1)

        fill_width = int(bar_width * intensity)
        color_intensity = (
            int(100 + 155 * intensity),
            int(255 - 100 * intensity),
            100
        )
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                      color_intensity, -1)

        cv2.putText(frame, f"Intensity: {int(intensity * 100)}%", (bar_x + bar_width + 20, bar_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Hand detection indicator
    hand_color = (100, 255, 100) if hand_detected else (80, 80, 80)
    hand_icon = "👋" if hand_detected else "🚫"
    cv2.putText(frame, f"{hand_icon} Hand: {'Detected' if hand_detected else 'Not Detected'}",
                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)

    # Recording indicator
    if recording_status:
        cv2.circle(frame, (w - 40, 60), 12, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 100, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

    # Controls hint
    cv2.putText(frame, "Controls: [R] Record | [Q] Quit",
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


def draw_gesture_guide(frame, locked, available_effects):
    """Draw gesture reference guide"""
    h, w = frame.shape[:2]
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
        instructions = ["Make gesture to LOCK mode:"]
        for effect in available_effects:
            instructions.append(f"{effect.icon} {effect.name}")

    y_offset = guide_y + 45
    for instruction in instructions:
        cv2.putText(frame, instruction, (guide_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y_offset += 22

    return frame
