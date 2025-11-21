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
    Detect QUIT gesture: Middle finger up only
    Returns: True if quit gesture detected
    """
    count, extended = count_extended_fingers(landmarks)

    # Middle finger up ONLY
    if extended[2] and not extended[0] and not extended[1] and not extended[3] and not extended[4]:
        return count == 1

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
