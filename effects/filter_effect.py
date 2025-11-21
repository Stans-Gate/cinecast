"""
Color Grade Effect
==================
Cinematic color grading with HSV manipulation
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect


class FilterEffect(BaseEffect):
    """Color grading filter"""

    def __init__(self):
        super().__init__(name="Color Grade", icon="👌", mode_id=4)

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply color grading
        intensity: 0.0 = original colors, 1.0 = full grade
        """
        if intensity < 0.05:
            return frame  # Skip if intensity too low

        # Convert to HSV for color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Cinematic color grade: warm shadows, cool highlights
        hsv[:, :, 0] = (hsv[:, :, 0] + intensity * 10) % 180  # Hue shift
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + intensity * 0.3), 0, 255)  # Saturation boost
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.9 + intensity * 0.2), 0, 255)  # Slight darkening

        hsv = hsv.astype(np.uint8)
        graded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend original with graded based on intensity
        output = cv2.addWeighted(frame, 1 - intensity, graded, intensity, 0)

        return output
