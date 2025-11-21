"""
[YOUR EFFECT NAME] Effect
=========================
[Brief description of what your effect does]

Template for creating new effects
----------------------------------
1. Copy this file and rename it (e.g., vignette_effect.py)
2. Update the class name (e.g., VignetteEffect)
3. Implement the apply() method
4. Add to effects/__init__.py
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect


class TemplateEffect(BaseEffect):
    """Your effect description"""

    def __init__(self):
        super().__init__(
            name="Your Effect Name",  # What shows in UI
            icon="🌟",                 # Emoji icon (pick any!)
            mode_id=999                # Change to next available ID (5, 6, 7...)
        )

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply your cinematic effect

        Args:
            frame: Input video frame (BGR format, numpy array)
            intensity: Effect strength (0.0 = no effect, 1.0 = maximum effect)
                      Controlled by user's palm openness
            time: Current time in seconds (useful for animations)

        Returns:
            Modified frame (BGR format, numpy array)
        """

        # Quick optimization: skip if intensity is very low
        if intensity < 0.05:
            return frame

        # Your effect code goes here!
        # Example: Simple brightness adjustment
        output = frame.copy()
        output = cv2.convertScaleAbs(output, alpha=1.0 + (intensity * 0.5), beta=0)

        return output

    def reset(self):
        """
        Optional: Reset any internal state when effect is deactivated
        Only implement if your effect needs to remember state between frames
        """
        pass


# Example effect ideas:
# =====================

# 1. Vignette (darken edges)
# ---------------------------
# mask = np.zeros_like(frame, dtype=np.float32)
# cv2.circle(mask, center, radius, (1, 1, 1), -1, cv2.LINE_AA)
# mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=100)
# output = (frame * mask).astype(np.uint8)

# 2. Film Grain
# -------------
# noise = np.random.randint(-intensity * 50, intensity * 50, frame.shape, dtype=np.int16)
# output = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# 3. Chromatic Aberration
# ------------------------
# shift = int(intensity * 10)
# b, g, r = cv2.split(frame)
# b_shifted = np.roll(b, -shift, axis=1)
# r_shifted = np.roll(r, shift, axis=1)
# output = cv2.merge([b_shifted, g, r_shifted])

# 4. Pixelate
# -----------
# scale = 1 - (intensity * 0.9)  # 1.0 to 0.1
# small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
# output = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

# 5. Edge Detection
# -----------------
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 150)
# edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
# output = cv2.addWeighted(frame, 1 - intensity, edges_color, intensity, 0)
