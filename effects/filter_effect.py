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
        Apply blue color filter
        intensity: 0.0 = original colors, 1.0 = full blue filter
        """
        if intensity < 0.05:
            return frame  # Skip if intensity too low

        # Create a blue tinted version of the frame
        blue_tinted = frame.copy().astype(np.float32)

        # Increase blue channel and reduce red/green for cool blue tone
        blue_tinted[:, :, 0] = np.clip(blue_tinted[:, :, 0] * (1.0 + intensity * 0.5), 0, 255)  # Blue channel boost
        blue_tinted[:, :, 1] = np.clip(blue_tinted[:, :, 1] * (1.0 - intensity * 0.3), 0, 255)  # Green reduction
        blue_tinted[:, :, 2] = np.clip(blue_tinted[:, :, 2] * (1.0 - intensity * 0.4), 0, 255)  # Red reduction

        blue_tinted = blue_tinted.astype(np.uint8)

        # Add slight desaturation for cinematic look
        hsv = cv2.cvtColor(blue_tinted, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - intensity * 0.2), 0, 255)  # Slight desaturation
        hsv = hsv.astype(np.uint8)
        graded = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend original with graded based on intensity
        output = cv2.addWeighted(frame, 1 - intensity, graded, intensity, 0)

        return output
