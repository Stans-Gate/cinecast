"""
UI Rendering Module
===================
All UI drawing functions
"""

import cv2
import numpy as np


def draw_ui(frame, effect_name, effect_icon, intensity, fps, recording_status, hand_detected, locked,
            gesture_name=None, gesture_confidence=0.0):
    """
    Draw comprehensive UI overlay
    Args:
        gesture_name: Recognized gesture name from MediaPipe (for debugging)
        gesture_confidence: Confidence score of the recognized gesture
    """
    h, w = frame.shape[:2]

    # Semi-transparent overlay panel (taller to fit gesture info)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Lock status
    if locked:
        lock_text = "[LOCK]"
        mode_color = (100, 255, 255)  # Cyan when locked
        status_text = "LOCKED"
    else:
        lock_text = "[UNLOCK]"
        mode_color = (100, 255, 100)  # Green when unlocked
        status_text = "UNLOCKED"

    # Mode display (skip emoji icon, just show name)
    cv2.putText(frame, f"{lock_text} {effect_name}", (20, 40),
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
    hand_status = "[HAND]" if hand_detected else "[NO HAND]"
    cv2.putText(frame, f"{hand_status} {'Detected' if hand_detected else 'Not Detected'}",
                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)

    # Gesture recognition output (for debugging)
    if gesture_name:
        gesture_color = (255, 200, 100)  # Orange/yellow for gesture info
        confidence_percent = int(gesture_confidence * 100)
        cv2.putText(frame, f"[GESTURE] {gesture_name} ({confidence_percent}%)",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 2)
    else:
        cv2.putText(frame, f"[GESTURE] None detected",
                    (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 2)

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


def draw_mode_menu(frame, available_effects, selected_index, menu_visible):
    """
    Draw mode selection menu on the right side of the screen
    Args:
        frame: Input frame
        available_effects: List of available effects/modes
        selected_index: Currently selected menu item index
        menu_visible: Whether menu should be visible
    Returns:
        Modified frame with menu overlay
    """
    if not menu_visible or len(available_effects) == 0:
        return frame
    
    h, w = frame.shape[:2]
    
    # Menu position and size
    menu_x = w - 300
    menu_y = 180
    menu_width = 280
    menu_item_height = 50
    max_visible_items = 6
    
    # Calculate visible range
    total_items = len(available_effects)
    visible_start = max(0, selected_index - max_visible_items // 2)
    visible_end = min(total_items, visible_start + max_visible_items)
    
    # Adjust start if we're near the end
    if visible_end - visible_start < max_visible_items:
        visible_start = max(0, visible_end - max_visible_items)
    
    # Semi-transparent menu background
    menu_height = min(max_visible_items, total_items) * menu_item_height + 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (menu_x, menu_y), (menu_x + menu_width, menu_y + menu_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Menu title
    cv2.putText(frame, "Mode Menu:", (menu_x + 10, menu_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw menu items
    y_offset = menu_y + 50
    for i in range(visible_start, visible_end):
        item_y = y_offset + (i - visible_start) * menu_item_height
        effect = available_effects[i]
        is_selected = (i == selected_index)
        
        # Highlight selected item
        if is_selected:
            cv2.rectangle(frame, (menu_x + 5, item_y - 20),
                         (menu_x + menu_width - 5, item_y + 25),
                         (100, 255, 255), 2)  # Cyan border for selected
        
        # Item background (slightly transparent)
        item_overlay = frame.copy()
        cv2.rectangle(item_overlay, (menu_x + 5, item_y - 20),
                     (menu_x + menu_width - 5, item_y + 25),
                     (40, 40, 40) if is_selected else (20, 20, 20), -1)
        cv2.addWeighted(item_overlay, 0.5, frame, 0.5, 0, frame)
        
        # Item text
        text_color = (255, 255, 255) if is_selected else (180, 180, 180)
        text_size = 0.55 if is_selected else 0.45
        
        # Name only (skip emoji icon)
        cv2.putText(frame, f"{effect.name}",
                   (menu_x + 15, item_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1 if is_selected else 1)
        
        # Draw selection indicator as a bullet point for selected item
        if is_selected:
            cv2.circle(frame, (menu_x + 12, item_y - 5), 3, text_color, -1)
    
    # Scroll indicators
    if visible_start > 0:
        cv2.putText(frame, "^", (menu_x + menu_width - 30, menu_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    if visible_end < total_items:
        cv2.putText(frame, "v", (menu_x + menu_width - 30, menu_y + menu_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    # Instructions
    instruction_y = menu_y + menu_height + 25
    cv2.putText(frame, "Index Finger Up/Down: Scroll", (menu_x + 10, instruction_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, "OK Sign: Select", (menu_x + 10, instruction_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return frame


def draw_gesture_guide(frame, locked, available_effects, current_effect=None):
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
        if current_effect and current_effect.mode_id == 5:  # 3D Object mode
            instructions = [
                "[LOCK] MODE (3D)",
                "",
                "Move hand: Rotate",
                "Palm open/close: Scale",
                "",
                "Thumbs Down = QUIT"
            ]
        else:
            instructions = [
                "[LOCK] MODE",
                "",
                "Close/Open palm = Intensity",
                "",
                "Thumbs Down = QUIT"
            ]
    else:
        # Show menu navigation instructions when unlocked
        instructions = [
            "[MENU] MODE",
            "",
            "Index Finger Up: Scroll Up",
            "Index Finger Down: Scroll Down",
            "OK Sign: Select Mode",
            "",
            "Use menu to choose mode"
        ]

    y_offset = guide_y + 45
    for instruction in instructions:
        cv2.putText(frame, instruction, (guide_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        y_offset += 22

    return frame


def draw_vfx_suggestion(frame, suggestion):
    """
    Draw VFX suggestion overlay on the frame

    Args:
        frame: Input frame
        suggestion: dict with keys:
            - effect_name: str
            - reason: str
            - confidence: float (0-1)
            - highlight_region: tuple (x, y, w, h) or None

    Returns:
        Modified frame with suggestion overlay
    """
    if not suggestion:
        return frame

    h, w = frame.shape[:2]

    # Draw highlight region if provided (e.g., detected book)
    if suggestion.get('highlight_region'):
        x, y, rect_w, rect_h = suggestion['highlight_region']

        # Draw pulsing highlight box
        import time
        pulse = 0.7 + 0.3 * abs(np.sin(time.time() * 3))

        # Green highlight for detected objects
        color = (int(100 * pulse), int(255 * pulse), int(100 * pulse))
        thickness = 3

        cv2.rectangle(frame, (x, y), (x + rect_w, y + rect_h), color, thickness)

        # Draw corner markers for emphasis
        marker_length = min(30, rect_w // 5, rect_h // 5)

        # Top-left corner
        cv2.line(frame, (x, y), (x + marker_length, y), color, thickness + 1)
        cv2.line(frame, (x, y), (x, y + marker_length), color, thickness + 1)

        # Top-right corner
        cv2.line(frame, (x + rect_w, y), (x + rect_w - marker_length, y), color, thickness + 1)
        cv2.line(frame, (x + rect_w, y), (x + rect_w, y + marker_length), color, thickness + 1)

        # Bottom-left corner
        cv2.line(frame, (x, y + rect_h), (x + marker_length, y + rect_h), color, thickness + 1)
        cv2.line(frame, (x, y + rect_h), (x, y + rect_h - marker_length), color, thickness + 1)

        # Bottom-right corner
        cv2.line(frame, (x + rect_w, y + rect_h), (x + rect_w - marker_length, y + rect_h), color, thickness + 1)
        cv2.line(frame, (x + rect_w, y + rect_h), (x + rect_w, y + rect_h - marker_length), color, thickness + 1)

    # Draw suggestion panel
    panel_width = 350
    panel_height = 100
    panel_x = (w - panel_width) // 2  # Center horizontally
    panel_y = h - panel_height - 30   # Bottom of screen

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Border with color based on confidence
    confidence = suggestion.get('confidence', 0)
    border_color = (
        int(100 + 155 * (1 - confidence)),  # B: less blue when confident
        int(100 + 155 * confidence),         # G: more green when confident
        int(100)                             # R: constant
    )
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  border_color, 2)

    # Title
    cv2.putText(frame, "VFX Suggestion",
                (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Effect name
    effect_name = suggestion.get('effect_name', 'Unknown')
    cv2.putText(frame, f"Effect: {effect_name}",
                (panel_x + 10, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

    # Reason
    reason = suggestion.get('reason', '')
    cv2.putText(frame, f"Reason: {reason}",
                (panel_x + 10, panel_y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Confidence bar
    bar_x = panel_x + 10
    bar_y = panel_y + 82
    bar_width = panel_width - 20
    bar_height = 10

    # Background bar
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (60, 60, 60), -1)

    # Confidence fill
    fill_width = int(bar_width * confidence)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_width, bar_y + bar_height),
                  border_color, -1)

    # Confidence percentage
    cv2.putText(frame, f"{int(confidence * 100)}%",
                (bar_x + bar_width + 5, bar_y + bar_height - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return frame
