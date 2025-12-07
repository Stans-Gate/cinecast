"""
Iron Man Gauntlet Effect
=========================
Virtual gauntlet and arc reactor on hand with dynamic lighting
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect
import math
import mediapipe as mp


class IronManEffect(BaseEffect):
    """Iron Man-inspired effect: gauntlet, arc reactor, and hand lighting"""

    def __init__(self):
        super().__init__(name="Iron Man Gauntlet", icon="✨", mode_id=6)
        
        # Arc reactor state
        self.arc_reactor_radius = 25
        self.arc_reactor_pulse = 0.0
        self.mp_hands = mp.solutions.hands
        self.hand_landmarks = None  # Store landmarks for apply()

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply Iron Man effect to frame
        intensity: Controls brightness of effect (0.0 to 1.0)
        Hand landmarks should be set via set_hand_landmarks() before calling
        """
        if self.hand_landmarks is None:
            return frame
            
        output = frame.copy()
        return self.draw_gauntlet_on_hand(output, self.hand_landmarks, intensity)

    def set_hand_landmarks(self, landmarks):
        """Set hand landmarks for the effect"""
        self.hand_landmarks = landmarks
    
    def draw_gauntlet_on_hand(self, frame, landmarks, intensity):
        """
        Draw gauntlet and arc reactor on hand
        Args:
            frame: Input frame
            landmarks: Hand landmarks array
            intensity: Palm openness (0.0 to 1.0)
        """
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Get palm center (middle of hand)
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Calculate palm center position in screen coordinates
        palm_x = int((wrist.x + middle_mcp.x) / 2 * w)
        palm_y = int((wrist.y + middle_mcp.y) / 2 * h)
        
        # Calculate palm orientation first
        palm_facing = self._calculate_palm_orientation(landmarks)
        
        # Show arc reactor based on palm facing and intensity
        # Lower threshold so light shows even when slightly turned
        show_reactor = palm_facing > 0.3 and intensity > 0.2
        
        if show_reactor:
            # Update arc reactor pulse
            self.arc_reactor_pulse += 0.1
            pulse_factor = 1.0 + 0.15 * math.sin(self.arc_reactor_pulse)
            base_radius = int(self.arc_reactor_radius * pulse_factor * intensity * (0.5 + 0.5 * palm_facing))
            current_radius = max(10, base_radius)
            
            # Create smooth glow using Gaussian blur approach with larger radius
            # Create a separate layer for the glow effect
            glow_layer = np.zeros((h, w, 3), dtype=np.float32)
            
            # Draw bright globule of light (not sun icon - just pure glow)
            # Create radial gradient for smooth glow - LARGER radius
            y_coords, x_coords = np.ogrid[:h, :w]
            distance_from_center = np.sqrt((x_coords - palm_x)**2 + (y_coords - palm_y)**2)
            
            # Larger glow radius with smooth falloff
            max_glow_radius = current_radius * 3.5  # Increased from 2.5 for larger radius
            glow_strength = np.exp(-distance_from_center / (max_glow_radius * 0.25))  # Softer falloff
            glow_strength = np.clip(glow_strength, 0, 1)
            
            # Create blue-white glow color (intensity based on palm facing)
            glow_color = np.array([1.0, 0.8, 0.6])  # BGR: cyan-blue to white
            glow_brightness = 0.6 + 0.4 * palm_facing  # Even when away, show some light
            
            # Apply glow with smooth gradient
            for c in range(3):
                glow_layer[:, :, c] = glow_color[2-c] * glow_strength * glow_brightness * 255
            
            # Apply Gaussian blur for smooth blending - larger sigma for smoother edges
            sigma = max(8, current_radius * 0.5)  # Larger blur for smoother edges
            glow_blurred = cv2.GaussianBlur(glow_layer.astype(np.uint8), (0, 0), sigmaX=sigma)
            
            # Bright core (inner bright globule - pure light, no icon)
            core_radius = int(current_radius * 0.5)
            # Draw bright white/cyan core with smooth edges
            for r in range(core_radius, 0, -2):
                alpha = r / core_radius
                color_intensity = int(255 * alpha)
                cv2.circle(glow_blurred, (palm_x, palm_y), r, 
                          (color_intensity, int(color_intensity * 0.95), int(color_intensity * 0.8)), -1)
            
            # Brightest center point
            cv2.circle(glow_blurred, (palm_x, palm_y), 3, (255, 255, 255), -1)
            
            # Blend glow layer - show even when palm slightly turned
            # Lower threshold: show light when palm_facing > 0.3
            if palm_facing > 0.3:
                # Show globule with intensity based on palm facing
                blend_alpha = 0.7 * (0.3 + 0.7 * palm_facing)  # Scale from 0.3 to 1.0
                output = cv2.addWeighted(output, 1.0 - blend_alpha, glow_blurred.astype(np.uint8), blend_alpha, 0)
            else:
                # Palm away - still show subtle light from palm
                blend_alpha = 0.3 * palm_facing  # Subtle light even when away
                output = cv2.addWeighted(output, 1.0 - blend_alpha, glow_blurred.astype(np.uint8), blend_alpha, 0)
        
        # Draw gauntlet lines on hand (always visible, but more prominent when palm facing)
        # Get key hand points
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        ring_mcp = landmarks[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Convert to screen coordinates
        index_mcp_x = int(index_mcp.x * w)
        index_mcp_y = int(index_mcp.y * h)
        pinky_mcp_x = int(pinky_mcp.x * w)
        pinky_mcp_y = int(pinky_mcp.y * h)
        ring_mcp_x = int(ring_mcp.x * w)
        ring_mcp_y = int(ring_mcp.y * h)
        middle_mcp_x = int(middle_mcp.x * w)
        middle_mcp_y = int(middle_mcp.y * h)
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        
        # Draw gauntlet outline - metallic gold/red Iron Man style
        # Visibility based on palm facing and intensity
        visibility = 0.6 + 0.4 * palm_facing
        gauntlet_base_color = (30, 80, 200)  # BGR: Gold-red metallic (darker, more visible)
        gauntlet_shadow_color = (15, 40, 100)  # Dark shadow for depth
        gauntlet_bright_color = (80, 140, 255)  # Brighter highlight
        base_thickness = 6 + int(5 * visibility * intensity)  # Much thicker, more visible
        
        # Main gauntlet arm structure (from wrist up) - with shadow for depth
        # Outer edge (index side) - draw shadow first, then main line
        cv2.line(output, (wrist_x + 1, wrist_y + 1), (index_mcp_x + 1, index_mcp_y + 1), 
                gauntlet_shadow_color, base_thickness + 2)
        cv2.line(output, (wrist_x, wrist_y), (index_mcp_x, index_mcp_y), gauntlet_base_color, base_thickness)
        cv2.line(output, (index_mcp_x, index_mcp_y), (palm_x, palm_y), gauntlet_base_color, base_thickness - 1)
        
        # Inner edge (pinky side)
        cv2.line(output, (wrist_x + 1, wrist_y + 1), (pinky_mcp_x + 1, pinky_mcp_y + 1), 
                gauntlet_shadow_color, base_thickness + 2)
        cv2.line(output, (wrist_x, wrist_y), (pinky_mcp_x, pinky_mcp_y), gauntlet_base_color, base_thickness)
        cv2.line(output, (pinky_mcp_x, pinky_mcp_y), (palm_x, palm_y), gauntlet_base_color, base_thickness - 1)
        
        # Add highlights for 3D metallic effect
        highlight_thickness = max(2, base_thickness // 2)
        cv2.line(output, (wrist_x, wrist_y), (index_mcp_x, index_mcp_y), gauntlet_bright_color, highlight_thickness)
        
        # Draw gauntlet plates/segments across knuckles
        cv2.line(output, (index_mcp_x, index_mcp_y), (middle_mcp_x, middle_mcp_y), gauntlet_base_color, base_thickness - 1)
        cv2.line(output, (middle_mcp_x, middle_mcp_y), (ring_mcp_x, ring_mcp_y), gauntlet_base_color, base_thickness - 1)
        cv2.line(output, (ring_mcp_x, ring_mcp_y), (pinky_mcp_x, pinky_mcp_y), gauntlet_base_color, base_thickness - 1)
        
        # Extend gauntlet lines to ALL fingers including thumb and pinky
        # Get finger tips for all fingers
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Convert to screen coordinates
        thumb_tip_x = int(thumb_tip.x * w)
        thumb_tip_y = int(thumb_tip.y * h)
        thumb_mcp_x = int(thumb_mcp.x * w)
        thumb_mcp_y = int(thumb_mcp.y * h)
        thumb_ip_x = int(thumb_ip.x * w)
        thumb_ip_y = int(thumb_ip.y * h)
        
        index_tip_x = int(index_tip.x * w)
        index_tip_y = int(index_tip.y * h)
        middle_tip_x = int(middle_tip.x * w)
        middle_tip_y = int(middle_tip.y * h)
        ring_tip_x = int(ring_tip.x * w)
        ring_tip_y = int(ring_tip.y * h)
        pinky_tip_x = int(pinky_tip.x * w)
        pinky_tip_y = int(pinky_tip.y * h)
        
        # Draw gauntlet lines extending from MCP to finger tips (all fingers)
        finger_thickness = max(2, base_thickness - 2)
        
        # THUMB - complete gauntlet line from wrist through thumb
        cv2.line(output, (wrist_x, wrist_y), (thumb_mcp_x, thumb_mcp_y), 
                gauntlet_base_color, base_thickness - 1)
        cv2.line(output, (thumb_mcp_x, thumb_mcp_y), (thumb_ip_x, thumb_ip_y), 
                gauntlet_base_color, finger_thickness)
        cv2.line(output, (thumb_ip_x, thumb_ip_y), (thumb_tip_x, thumb_tip_y), 
                gauntlet_base_color, finger_thickness - 1)
        cv2.line(output, (wrist_x + 1, wrist_y + 1), (thumb_mcp_x + 1, thumb_mcp_y + 1), 
                gauntlet_shadow_color, base_thickness)
        
        # INDEX finger (outer edge)
        cv2.line(output, (index_mcp_x, index_mcp_y), (index_tip_x, index_tip_y), 
                gauntlet_base_color, finger_thickness)
        cv2.line(output, (index_mcp_x + 1, index_mcp_y + 1), (index_tip_x + 1, index_tip_y + 1), 
                gauntlet_shadow_color, finger_thickness + 1)
        
        # MIDDLE finger (center)
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        middle_pip_x = int(middle_pip.x * w)
        middle_pip_y = int(middle_pip.y * h)
        cv2.line(output, (middle_mcp_x, middle_mcp_y), (middle_pip_x, middle_pip_y), 
                gauntlet_base_color, finger_thickness)
        cv2.line(output, (middle_pip_x, middle_pip_y), (middle_tip_x, middle_tip_y), 
                gauntlet_base_color, finger_thickness - 1)
        
        # RING finger (between middle and pinky)
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        ring_pip_x = int(ring_pip.x * w)
        ring_pip_y = int(ring_pip.y * h)
        cv2.line(output, (ring_mcp_x, ring_mcp_y), (ring_pip_x, ring_pip_y), 
                gauntlet_base_color, finger_thickness)
        cv2.line(output, (ring_pip_x, ring_pip_y), (ring_tip_x, ring_tip_y), 
                gauntlet_base_color, finger_thickness - 1)
        
        # PINKY finger (inner edge) - complete line to tip
        cv2.line(output, (pinky_mcp_x, pinky_mcp_y), (pinky_tip_x, pinky_tip_y), 
                gauntlet_base_color, finger_thickness)
        cv2.line(output, (pinky_mcp_x + 1, pinky_mcp_y + 1), (pinky_tip_x + 1, pinky_tip_y + 1), 
                gauntlet_shadow_color, finger_thickness + 1)
        
        # Add decorative elements - circles at joints for tech look
        joint_size = 6
        cv2.circle(output, (wrist_x, wrist_y), joint_size, gauntlet_shadow_color, -1)
        cv2.circle(output, (wrist_x, wrist_y), joint_size - 2, gauntlet_bright_color, -1)
        cv2.circle(output, (index_mcp_x, index_mcp_y), joint_size - 1, gauntlet_bright_color, -1)
        cv2.circle(output, (pinky_mcp_x, pinky_mcp_y), joint_size - 1, gauntlet_bright_color, -1)
        cv2.circle(output, (middle_mcp_x, middle_mcp_y), joint_size - 2, gauntlet_bright_color, -1)
        cv2.circle(output, (thumb_mcp_x, thumb_mcp_y), joint_size - 1, gauntlet_bright_color, -1)
        
        # Small circles at finger tips (all fingers)
        cv2.circle(output, (thumb_tip_x, thumb_tip_y), 3, gauntlet_bright_color, -1)
        cv2.circle(output, (index_tip_x, index_tip_y), 3, gauntlet_bright_color, -1)
        cv2.circle(output, (middle_tip_x, middle_tip_y), 3, gauntlet_bright_color, -1)
        cv2.circle(output, (ring_tip_x, ring_tip_y), 3, gauntlet_bright_color, -1)
        cv2.circle(output, (pinky_tip_x, pinky_tip_y), 3, gauntlet_bright_color, -1)
        
        # Apply dynamic lighting based on palm orientation and distance
        # Always apply lighting (even when palm away) but with different intensity
        
        # Create radial brightness gradient from palm center
        y_coords, x_coords = np.ogrid[:h, :w]
        distance_from_palm = np.sqrt((x_coords - palm_x)**2 + (y_coords - palm_y)**2)
        max_light_distance = min(w, h) * 0.7  # Light reaches further when palm facing
        
        # Lighting intensity based on palm orientation
        # Keep same ambient darkness level regardless of palm orientation
        base_darkness = 0.75  # Same minimum darkness for both cases
        
        if palm_facing > 0.5:
            # Palm facing: bright area around palm, same ambient darkness
            lighting_intensity = 0.3 + 0.7 * intensity  # Strong lighting
            light_falloff = distance_from_palm / (max_light_distance * lighting_intensity)
            light_mask = np.exp(-light_falloff * 2.5)  # Sharp falloff
            
            # Brighten center area, but keep same minimum darkness for far areas
            light_mask = light_mask * lighting_intensity * 1.5  # Reduced from 1.6
            # Don't darken far areas - keep same ambient level
            light_mask = np.clip(light_mask, base_darkness, 1.8)  # Same minimum as palm away
        else:
            # Palm away: still show lighting behind hand with same ambient darkness
            lighting_intensity = 0.25 + 0.4 * intensity  # Moderate lighting
            light_falloff = distance_from_palm / (max_light_distance * max(0.5, lighting_intensity))
            light_mask = np.exp(-light_falloff * 2.0)  # Softer falloff
            
            # Subtle lighting with same ambient darkness
            light_mask = light_mask * lighting_intensity * 1.0  # Increased from 0.8
            light_mask = np.clip(light_mask, base_darkness, 1.0)  # Same minimum darkness
        
        # Apply lighting - brighten area around palm (or behind when palm away)
        output_float = output.astype(np.float32)
        light_mask_3d = light_mask[:, :, np.newaxis]
        
        # Apply lighting effect
        brightened = output_float * light_mask_3d
        brightened = np.clip(brightened, 0, 255)
        
        output = brightened.astype(np.uint8)
        
        return output
    
    def _calculate_palm_orientation(self, landmarks):
        """
        Calculate if palm is facing camera (1 = facing, 0 = away)
        Returns: 0.0 to 1.0
        Simple method: Check thumb tip position relative to wrist
        Right hand: thumb to right = palm facing, thumb to left = palm away
        """
        # Get thumb tip and wrist positions
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Simple check: thumb tip x-position relative to wrist
        # For right hand: thumb tip to the right of wrist = palm facing camera
        thumb_relative_x = thumb_tip.x - wrist.x
        
        # Normalize to 0-1 range with better handling for slightly turned hands
        # Typical range: -0.25 to 0.25
        # When palm facing (right hand): thumb is to right, so positive value
        # When palm away: thumb is to left, so negative value
        
        # Use wider range and smoother transition for slightly turned hands
        # This allows light to show even when hand is slightly turned but thumb still to right
        orientation = (thumb_relative_x + 0.25) / 0.5  # Map -0.25 to 0.25 -> 0 to 1
        orientation = max(0.0, min(1.0, orientation))  # Clamp to 0-1
        
        # Apply smooth curve for better sensitivity near threshold
        orientation = orientation ** 0.7  # Gentle curve - less harsh transition
        
        return orientation

    def reset(self):
        """Reset effect state"""
        self.arc_reactor_pulse = 0.0

