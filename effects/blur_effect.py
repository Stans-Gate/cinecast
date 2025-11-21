"""
Motion Blur Effect
==================
Variable blur intensity for focus/defocus effects
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect


class BlurEffect(BaseEffect):
    """Motion blur / defocus effect"""

    def __init__(self):
        super().__init__(name="Motion Blur", icon="🤘", mode_id=3)

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply blur effect
        intensity: 0.0 = sharp, 1.0 = maximum blur
        """
        if intensity < 0.05:
            return frame  # Skip blur if intensity too low

        # Map intensity to kernel size (must be odd)
        kernel_size = int(5 + intensity * 46)  # 5 to 51
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel_size = max(5, min(51, kernel_size))

        # Apply Gaussian blur
        output = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        return output
