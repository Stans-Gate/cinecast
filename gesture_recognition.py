"""
Gesture Recognition Module
===========================
All hand gesture detection logic
"""

import mediapipe as mp
import numpy as np
import math


mp_hands = mp.solutions.hands


def count_extended_fingers(landmarks):
    """
    Count how many fingers are extended
    Returns: (count, [thumb, index, middle, ring, pinky])
    """
    fingers_extended = [False] * 5

    # Thumb: check if tip is far from palm in x-direction
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]

    thumb_tip_dist = abs(thumb_tip.x - wrist.x)
    thumb_ip_dist = abs(thumb_ip.x - wrist.x)
    fingers_extended[0] = thumb_tip_dist > thumb_ip_dist * 1.3

    # Other fingers: tip above PIP joint
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    finger_mcps = [5, 9, 13, 17]

    for i, (tip, pip, mcp) in enumerate(zip(finger_tips, finger_pips, finger_mcps)):
        tip_y = landmarks[tip].y
        pip_y = landmarks[pip].y
        mcp_y = landmarks[mcp].y

        is_extended = (tip_y < pip_y - 0.02) and (tip_y < mcp_y)
        fingers_extended[i + 1] = is_extended

    return sum(fingers_extended), fingers_extended


def get_palm_openness(landmarks):
    """
    Calculate palm openness based on finger curl
    Returns: 0.0 (closed fist) to 1.0 (open palm)
    """
    count, extended = count_extended_fingers(landmarks)

    # Calculate openness as ratio of extended fingers
    openness = count / 5.0

    # Add granularity by checking finger curl distances
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]

    curl_distances = []
    for tip, mcp in zip(finger_tips, finger_mcp):
        tip_pos = landmarks[tip]
        mcp_pos = landmarks[mcp]
        dist = ((tip_pos.x - mcp_pos.x)**2 + (tip_pos.y - mcp_pos.y)**2) ** 0.5
        curl_distances.append(dist)

    avg_curl = np.mean(curl_distances)

    # Normalize curl distance
    curl_intensity = (avg_curl - 0.05) / (0.15 - 0.05)
    curl_intensity = np.clip(curl_intensity, 0.0, 1.0)

    # Blend both metrics
    final_openness = 0.6 * curl_intensity + 0.4 * openness

    return np.clip(final_openness, 0.0, 1.0)


def is_quit_gesture(landmarks):
    """
    Detect QUIT gesture: Fist (all fingers closed, thumb wrapped)
    Returns: True if quit gesture detected
    """
    count, extended = count_extended_fingers(landmarks)

    # Fist: all fingers closed (no fingers extended)
    if count == 0:
        # Additional check: thumb should be wrapped (thumb tip below thumb IP or close)
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
        
        # For a proper fist, thumb tip should be close to or below thumb IP
        thumb_wrapped = thumb_tip.y >= thumb_ip.y - 0.03
        return thumb_wrapped
    
    return False


def classify_mode_gesture(landmarks, available_effects):
    """
    Classify gesture to lock into a mode
    Returns: effect mode_id or None
    """
    count, extended = count_extended_fingers(landmarks)

    # Get thumb and index positions for OK sign
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ok_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) ** 0.5

    # OK SIGN (mode_id 4)
    if ok_dist < 0.06 and (extended[2] or extended[3] or extended[4]):
        others_up = sum([extended[2], extended[3], extended[4]])
        if others_up >= 2:
            return 4

    # THUMBS UP (mode_id 1)
    if extended[0] and count == 1:
        return 1

    # PEACE SIGN (mode_id 2)
    if extended[1] and extended[2] and not extended[3] and not extended[4]:
        if count == 2 or (count == 3 and extended[0]):
            return 2

    # ROCK SIGN (mode_id 3)
    if extended[1] and extended[4] and not extended[2] and not extended[3]:
        if count == 2 or (count == 3 and extended[0]):
            return 3

    return None


def detect_menu_scroll_gesture(landmarks):
    """
    Detect menu scroll gesture: index finger pointing up (scroll up) or down (scroll down)
    Returns: 'up', 'down', or None
    """
    count, extended = count_extended_fingers(landmarks)
    
    # Get index finger landmarks
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_dip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    
    # Check if index finger is straight/extended regardless of direction
    # Measure distance from tip to MCP to determine if finger is extended
    tip_to_mcp_dist = ((index_tip.x - index_mcp.x)**2 + (index_tip.y - index_mcp.y)**2) ** 0.5
    pip_to_mcp_dist = ((index_pip.x - index_mcp.x)**2 + (index_pip.y - index_mcp.y)**2) ** 0.5
    
    # Finger is extended if tip is far from MCP (relative to PIP distance)
    # This works regardless of whether pointing up or down
    is_extended = tip_to_mcp_dist > pip_to_mcp_dist * 1.5
    
    if not is_extended:
        return None
    
    # Check index finger orientation (up or down)
    # Index finger pointing up: tip is above PIP and MCP
    index_pointing_up = (index_tip.y < index_pip.y - 0.02 and 
                        index_tip.y < index_mcp.y - 0.02)
    
    # Index finger pointing down: tip is below PIP and MCP
    index_pointing_down = (index_tip.y > index_pip.y + 0.02 and 
                          index_tip.y > index_mcp.y + 0.02)
    
    # For scroll up: index pointing up, other fingers mostly closed
    if index_pointing_up:
        # Allow thumb to be extended, but other fingers should be closed
        other_fingers_closed = not extended[2] and not extended[3] and not extended[4]
        if other_fingers_closed:
            return 'up'
    
    # For scroll down: index pointing down, other fingers mostly closed
    elif index_pointing_down:
        # Allow thumb to be extended, but other fingers should be closed
        other_fingers_closed = not extended[2] and not extended[3] and not extended[4]
        if other_fingers_closed:
            return 'down'
    
    return None


def detect_menu_select_gesture(landmarks):
    """
    Detect menu selection gesture: OK sign (thumb and index finger touching)
    Returns: True if selection gesture detected
    """
    count, extended = count_extended_fingers(landmarks)
    
    # Get thumb and index positions
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Check distance between thumb and index tip (OK sign)
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2) ** 0.5
    
    # OK sign: thumb and index touching (or very close), other fingers extended
    if distance < 0.05:  # Thumb and index are touching/close
        # Other fingers should be extended
        others_extended = extended[2] or extended[3] or extended[4]
        if others_extended or count >= 2:
            return True
    
    return False


def is_quit_gesture(landmarks):
    """
    Detect QUIT gesture: Thumbs down (thumb extended downward)
    Returns: True if quit gesture detected
    """
    count, extended = count_extended_fingers(landmarks)
    
    # Get thumb landmarks
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    
    # Thumbs down: thumb extended downward (tip below MCP in Y direction)
    # And thumb is extended outward from hand
    thumb_extended_outward = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 1.2
    thumb_pointing_down = thumb_tip.y > thumb_mcp.y + 0.05  # Thumb tip below MCP
    
    # All other fingers should be closed (or mostly closed)
    other_fingers_closed = not extended[1] and not extended[2] and not extended[3] and not extended[4]
    
    if thumb_extended_outward and thumb_pointing_down and other_fingers_closed:
        return True
    
    return False


def get_hand_position(landmarks):
    """
    Get normalized hand position (for 3D interaction)
    Returns: (x, y) tuple where values are 0.0 to 1.0
    """
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    return (wrist.x, wrist.y)


def detect_3d_rotation_gesture(landmarks, previous_hand_angles=None):
    """
    Calculate hand rotation angles directly from hand orientation
    Returns: (angle_x, angle_y) tuple in radians, or (None, None) if not available
    Direct mapping: hand orientation -> object rotation
    Uses more stable landmarks and better angle calculation
    """
    # Calculate hand orientation from key landmarks
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    
    # Use more stable vectors - from wrist to middle finger PIP (more stable than MCP)
    # This gives better angle stability
    hand_up_x = middle_pip.x - wrist.x
    hand_up_y = middle_pip.y - wrist.y
    
    # Use index finger for horizontal orientation (more stable)
    hand_right_x = index_mcp.x - wrist.x
    hand_right_y = index_mcp.y - wrist.y
    
    # Calculate magnitudes to check if hand is in valid pose
    hand_up_mag = math.sqrt(hand_up_x**2 + hand_up_y**2)
    hand_right_mag = math.sqrt(hand_right_x**2 + hand_right_y**2)
    
    # Only proceed if hand is in a valid pose (fingers extended enough)
    if hand_up_mag < 0.05 or hand_right_mag < 0.03:
        return (None, None)  # Hand too closed or invalid pose
    
    # Calculate hand rotation angles with better stability
    # X rotation (pitch): tilt hand forward/back
    # Use normalized vector for more stable angle calculation
    angle_x = -math.atan2(hand_up_y, hand_up_mag + 0.05)  # Add small epsilon for stability
    
    # Y rotation (yaw): tilt hand left/right
    angle_y = math.atan2(hand_right_x, hand_right_mag + 0.05)
    
    # Clamp angles to reasonable range (prevent extreme values)
    angle_x = max(-math.pi/2, min(math.pi/2, angle_x))  # Limit to ±90 degrees
    angle_y = max(-math.pi/2, min(math.pi/2, angle_y))
    
    return (angle_x, angle_y)


def detect_3d_scale_gesture(landmarks):
    """
    Detect 3D scale gesture based on palm openness
    Returns: scale value (1.0 = normal, >1.0 = zoom out, <1.0 = zoom in)
    """
    openness = get_palm_openness(landmarks)
    
    # Map palm openness to scale: closed fist = zoom in (0.5x), open palm = zoom out (1.5x)
    # Inverse mapping: closed (0.0) -> zoom in, open (1.0) -> zoom out
    scale = 0.5 + openness * 1.0  # Range: 0.5 to 1.5
    
    return scale
