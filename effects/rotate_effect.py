"""
Rotate Effect
=============
Smooth continuous rotation based on intensity
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect


class RotateEffect(BaseEffect):
    """Continuous rotation effect"""

    def __init__(self):
        super().__init__(name="Rotate", icon="✌️", mode_id=2)

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply rotation effect
        intensity: 0.0 = no rotation, 1.0 = fast rotation (30°/sec)
        """
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        # Rotation speed based on intensity
        angle = (time * 30 * intensity) % 360

        # Apply rotation transformation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        output = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return output
