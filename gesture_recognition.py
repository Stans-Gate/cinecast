"""
Gesture Recognition Module
===========================
All hand gesture detection logic
"""

import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands


def count_extended_fingers(landmarks):
    """
    Count how many fingers are extended
    Returns: (count, [thumb, index, middle, ring, pinky])
    """
    fingers_extended = [False] * 5

    # Thumb: check if tip is extended both horizontally and vertically
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]

    # Calculate distance from tip to wrist vs IP to wrist
    thumb_tip_dist = ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2) ** 0.5
    thumb_ip_dist = ((thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2) ** 0.5

    # Also check if thumb is above (lower y value) the MCP joint
    thumb_is_up = thumb_tip.y < thumb_mcp.y

    # Thumb is extended if tip is farther from wrist than IP joint
    fingers_extended[0] = (thumb_tip_dist > thumb_ip_dist * 1.2) or thumb_is_up

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
    Detect QUIT gesture: Pinky finger up only (like "call me" gesture)
    Returns: True if quit gesture detected
    """
    count, extended = count_extended_fingers(landmarks)

    # Pinky finger up ONLY (index 4 in the extended array)
    # Accept with or without thumb extended
    if extended[4] and not extended[1] and not extended[2] and not extended[3]:
        # Either just pinky, or pinky + thumb (more forgiving)
        if count == 1 or (count == 2 and extended[0]):
            return True

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
