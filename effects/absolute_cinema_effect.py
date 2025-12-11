"""
Absolute Cinema Effect
=======================
Epic cinematic moment with black & white conversion and bold text overlay.
Triggered by raising both hands with open palms.
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect


class AbsoluteCinemaEffect(BaseEffect):
    """Absolute Cinema - Epic B&W effect with dramatic text"""

    def __init__(self):
        super().__init__(
            name="Absolute Cinema",
            icon="🎬",
            mode_id=7
        )
        self.activation_time = None

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply the Absolute Cinema effect

        Args:
            frame: Input frame (BGR)
            intensity: 0.0 to 1.0 (controls fade-in/fade-out)
            time: Current time in seconds

        Returns:
            Modified frame with B&W and text overlay
        """
        h, w = frame.shape[:2]
        output = frame.copy()

        # Track activation time for animations
        if self.activation_time is None:
            self.activation_time = time

        elapsed = time - self.activation_time

        # Convert to black and white
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Apply cinematic contrast (increase contrast for dramatic effect)
        alpha = 1.5  # Contrast
        beta = -30   # Brightness
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

        # Add vignette effect for cinematic feel
        output = self._add_vignette(output, intensity)

        # Calculate text animation parameters
        # Text fades in over 0.5 seconds, stays visible, fades out when intensity drops
        fade_in_duration = 0.5
        if elapsed < fade_in_duration:
            text_alpha = elapsed / fade_in_duration
        else:
            text_alpha = intensity

        # Draw "ABSOLUTE CINEMA" text
        if text_alpha > 0.01:
            output = self._draw_text_overlay(output, text_alpha, time)

        # Blend with original based on intensity for smooth transitions
        if intensity < 1.0:
            output = cv2.addWeighted(frame, 1 - intensity, output, intensity, 0)

        return output

    def _add_vignette(self, frame, intensity):
        """Add cinematic vignette effect"""
        h, w = frame.shape[:2]

        # Create radial gradient for vignette
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]

        # Calculate distance from center
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)

        # Create vignette mask (darker at edges)
        vignette = 1 - (dist_from_center / max_dist) ** 2
        vignette = np.clip(vignette, 0.3, 1.0)  # Don't make edges completely black

        # Apply vignette with intensity control
        vignette_strength = 0.5 + intensity * 0.5  # More vignette as intensity increases
        vignette = 1 - (1 - vignette) * vignette_strength

        # Apply to frame
        vignette = vignette[:, :, np.newaxis]
        output = (frame * vignette).astype(np.uint8)

        return output

    def _draw_text_overlay(self, frame, alpha, time):
        """Draw the ABSOLUTE CINEMA text overlay"""
        h, w = frame.shape[:2]

        # Create overlay for text
        overlay = frame.copy()

        # Text settings
        text = "ABSOLUTE CINEMA"
        font = cv2.FONT_HERSHEY_DUPLEX  # Bold-looking font

        # Calculate font scale based on frame size
        base_scale = w / 800  # Scale based on width
        font_scale = 2.5 * base_scale
        thickness = int(8 * base_scale)

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate position (center of frame)
        text_x = (w - text_width) // 2
        text_y = (h + text_height) // 2

        # Add subtle animation - slight pulsing
        pulse = 1.0 + 0.05 * np.sin(time * 3)
        animated_scale = font_scale * pulse
        animated_thickness = int(thickness * pulse)

        # Recalculate with animated scale
        (text_width, text_height), baseline = cv2.getTextSize(text, font, animated_scale, animated_thickness)
        text_x = (w - text_width) // 2
        text_y = (h + text_height) // 2

        # Draw text shadow/outline for better visibility
        shadow_offset = int(5 * base_scale)
        cv2.putText(overlay, text, (text_x + shadow_offset, text_y + shadow_offset),
                   font, animated_scale, (0, 0, 0), animated_thickness + 2, cv2.LINE_AA)

        # Draw main text in white
        cv2.putText(overlay, text, (text_x, text_y),
                   font, animated_scale, (255, 255, 255), animated_thickness, cv2.LINE_AA)

        # Add cinematic bars (letterbox effect)
        bar_height = int(h * 0.15)
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)

        # Blend overlay with frame using alpha
        output = cv2.addWeighted(frame, 1 - alpha * 0.9, overlay, alpha * 0.9, 0)

        return output

    def reset(self):
        """Reset effect state"""
        self.activation_time = None
