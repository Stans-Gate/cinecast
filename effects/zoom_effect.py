"""
Dolly Zoom Effect
=================
Cinematic zoom effect with subtle oscillation
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect


class ZoomEffect(BaseEffect):
    """Dolly zoom effect - zoom in/out based on intensity"""

    def __init__(self):
        super().__init__(name="Dolly Zoom", icon="👍", mode_id=1)

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply zoom effect
        intensity: 0.0 = zoom out (0.7x), 1.0 = zoom in (1.5x)
        """
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)

        # Map intensity to scale factor
        scale = 0.7 + intensity * 0.8  # 0.7x to 1.5x

        # Add subtle oscillation for cinematic feel
        scale += 0.05 * np.sin(time * 2.0) * intensity

        # Apply zoom transformation
        M = cv2.getRotationMatrix2D(center, 0, scale)
        output = cv2.warpAffine(
            frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return output
