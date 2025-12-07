"""
Iron Man Gauntlet Effect
========================
Creates an Iron-Man-inspired AR gauntlet overlay with dynamic scene relighting.

Features:
- Gold/red armor plating overlay on hand
- Glowing arc reactor in palm
- Dynamic scene relighting based on palm orientation and openness
- Brightens area around palm, darkens rest of scene
"""

from effects.base_effect import BaseEffect
import cv2
import numpy as np


class IronManEffect(BaseEffect):
    def __init__(self):
        super().__init__(
            name="Iron Man Gauntlet",
            icon="🦾",
            mode_id=6
        )
        # Import mediapipe lazily to avoid import errors
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
        except ImportError:
            self.mp_hands = None

    def apply(self, frame, intensity, time):
        """
        Apply Iron Man gauntlet effect with dynamic relighting

        Args:
            frame: Input frame (BGR)
            intensity: 0.0 to 1.0 (controlled by palm openness)
            time: Current time in seconds

        Returns:
            Modified frame with gauntlet overlay and relighting
        """
        if intensity < 0.05:
            return frame

        # Check if mediapipe is available
        if self.mp_hands is None:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands

        h, w = frame.shape[:2]
        output = frame.copy()

        # Detect hand landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        import mediapipe as mp
        hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        results = hands.process(rgb)
        hands.close()

        if not results.multi_hand_landmarks:
            return frame

        landmarks = results.multi_hand_landmarks[0].landmark

        # Get palm center and orientation
        palm_center, palm_orientation = self._get_palm_info(landmarks, w, h)

        # Apply dynamic relighting FIRST (background layer)
        output = self._apply_relighting(output, palm_center, palm_orientation, intensity, time)

        # Draw gauntlet overlay (foreground layer)
        output = self._draw_gauntlet(output, landmarks, w, h, intensity)

        # Draw arc reactor glow
        output = self._draw_arc_reactor(output, palm_center, intensity, time)

        return output

    def _get_palm_info(self, landmarks, w, h):
        """
        Calculate palm center position and orientation

        Returns:
            palm_center: (x, y) tuple in pixel coordinates
            palm_orientation: 0.0 to 1.0 (1.0 = facing camera)
        """
        # Palm center: average of wrist, index base, pinky base
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        index_mcp = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = landmarks[self.mp_hands.HandLandmark.PINKY_MCP]
        middle_mcp = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

        palm_x = int((wrist.x + index_mcp.x + pinky_mcp.x) / 3 * w)
        palm_y = int((wrist.y + index_mcp.y + pinky_mcp.y) / 3 * h)
        palm_center = (palm_x, palm_y)

        # Calculate palm normal vector using cross product
        # Vector 1: wrist -> middle MCP
        v1 = np.array([
            middle_mcp.x - wrist.x,
            middle_mcp.y - wrist.y,
            middle_mcp.z - wrist.z
        ])

        # Vector 2: pinky MCP -> index MCP (across palm)
        v2 = np.array([
            index_mcp.x - pinky_mcp.x,
            index_mcp.y - pinky_mcp.y,
            index_mcp.z - pinky_mcp.z
        ])

        # Normal vector (perpendicular to palm)
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-6)

        # Camera is looking down negative Z axis
        # Dot product with camera direction tells us orientation
        camera_dir = np.array([0, 0, -1])
        orientation = np.dot(normal, camera_dir)

        # Map to 0-1 range (0 = away, 1 = toward camera)
        orientation = np.clip((orientation + 1) / 2, 0, 1)

        return palm_center, orientation

    def _apply_relighting(self, frame, palm_center, palm_orientation, intensity, time):
        """
        Apply dynamic scene relighting based on palm position and orientation
        Creates the effect of the arc reactor emitting light into the scene
        """
        h, w = frame.shape[:2]

        # Create distance map from palm center
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - palm_center[0])**2 + (y_coords - palm_center[1])**2)

        # Light intensity based on distance, orientation, and palm openness
        # Combine all factors
        light_power = intensity * palm_orientation * 1.5

        # Inverse square falloff with adjustable radius
        light_radius = 300 * (0.5 + intensity * 0.5)  # Radius grows with intensity
        light_mask = light_power / (1 + (distances / light_radius) ** 2)
        light_mask = np.clip(light_mask, 0, 1)

        # Create illuminated and darkened regions
        # Brighten near palm (additive)
        brighten_mask = light_mask ** 0.8  # Softer falloff for brightening
        brighten_mask = brighten_mask[:, :, np.newaxis]

        # Darken rest of scene (multiplicative vignette)
        darken_factor = 1 - light_power * 0.6  # Max 60% darkening
        vignette = 1 - (1 - light_mask) * (1 - darken_factor)
        vignette = vignette[:, :, np.newaxis]

        # Arc reactor color (cyan-blue)
        reactor_color = np.array([255, 200, 100], dtype=np.float32)  # BGR: cyan-blue

        # Add pulsing effect
        pulse = 0.85 + 0.15 * np.sin(time * 8)
        reactor_color = reactor_color * pulse

        # Apply lighting
        # First darken the scene
        output = (frame.astype(np.float32) * vignette).astype(np.uint8)

        # Then add illumination with reactor color
        illumination = brighten_mask * reactor_color * 0.6
        output = cv2.add(output, illumination.astype(np.uint8))

        return output

    def _draw_gauntlet(self, frame, landmarks, w, h, intensity):
        """
        Draw stylized armor plating over the hand
        """
        # Create overlay for transparency
        overlay = frame.copy()

        # Get landmark positions
        def get_point(landmark_id):
            lm = landmarks[landmark_id]
            return (int(lm.x * w), int(lm.y * h))

        # Armor colors (gold/red with metallic look)
        gold_base = (0, 140, 218)  # BGR: gold
        gold_highlight = (100, 200, 255)  # Lighter gold
        red_accent = (30, 30, 200)  # Red
        dark_edge = (0, 50, 100)  # Dark edges

        alpha = intensity * 0.7  # Transparency

        # Draw palm armor plate (main piece)
        wrist = get_point(self.mp_hands.HandLandmark.WRIST)
        index_mcp = get_point(self.mp_hands.HandLandmark.INDEX_FINGER_MCP)
        middle_mcp = get_point(self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
        ring_mcp = get_point(self.mp_hands.HandLandmark.RING_FINGER_MCP)
        pinky_mcp = get_point(self.mp_hands.HandLandmark.PINKY_MCP)

        palm_points = np.array([
            wrist,
            pinky_mcp,
            ring_mcp,
            middle_mcp,
            index_mcp
        ], dtype=np.int32)

        cv2.fillPoly(overlay, [palm_points], gold_base)
        cv2.polylines(overlay, [palm_points], True, dark_edge, 3)

        # Draw finger armor segments
        finger_tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]

        finger_pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        finger_mcps = [
            self.mp_hands.HandLandmark.THUMB_CMC,
            self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            self.mp_hands.HandLandmark.RING_FINGER_MCP,
            self.mp_hands.HandLandmark.PINKY_MCP
        ]

        # Draw armor segments on each finger
        for tip_id, pip_id, mcp_id in zip(finger_tips, finger_pips, finger_mcps):
            tip = get_point(tip_id)
            pip = get_point(pip_id)
            mcp = get_point(mcp_id)

            # Calculate perpendicular offset for segment width
            dx = tip[0] - mcp[0]
            dy = tip[1] - mcp[1]
            length = np.sqrt(dx**2 + dy**2) + 1e-6
            perp_x = -dy / length * 8
            perp_y = dx / length * 8

            # Upper segment (tip to pip)
            segment1 = np.array([
                [tip[0] + int(perp_x), tip[1] + int(perp_y)],
                [tip[0] - int(perp_x), tip[1] - int(perp_y)],
                [pip[0] - int(perp_x), pip[1] - int(perp_y)],
                [pip[0] + int(perp_x), pip[1] + int(perp_y)]
            ], dtype=np.int32)

            cv2.fillPoly(overlay, [segment1], red_accent)
            cv2.polylines(overlay, [segment1], True, dark_edge, 2)

            # Lower segment (pip to mcp)
            segment2 = np.array([
                [pip[0] + int(perp_x), pip[1] + int(perp_y)],
                [pip[0] - int(perp_x), pip[1] - int(perp_y)],
                [mcp[0] - int(perp_x), mcp[1] - int(perp_y)],
                [mcp[0] + int(perp_x), mcp[1] + int(perp_y)]
            ], dtype=np.int32)

            cv2.fillPoly(overlay, [segment2], gold_highlight)
            cv2.polylines(overlay, [segment2], True, dark_edge, 2)

        # Blend overlay with original frame
        output = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        return output

    def _draw_arc_reactor(self, frame, palm_center, intensity, time):
        """
        Draw glowing arc reactor in palm with pulsing animation
        """
        overlay = frame.copy()

        # Pulsing animation
        pulse = 0.7 + 0.3 * np.sin(time * 6)

        # Arc reactor colors
        core_color = (255, 200, 50)  # Bright cyan
        ring_color = (255, 150, 0)  # Lighter cyan
        outer_glow = (200, 100, 0)  # Soft blue glow

        # Base reactor size
        base_radius = int(25 + intensity * 15)

        # Draw outer glow (largest, most transparent)
        for i in range(5, 0, -1):
            radius = int(base_radius * (1 + i * 0.4) * pulse)
            alpha = intensity * 0.05 * i
            cv2.circle(overlay, palm_center, radius, outer_glow, -1)

        # Draw concentric rings
        num_rings = 3
        for i in range(num_rings):
            ring_radius = int(base_radius * (1 - i * 0.25))
            thickness = max(2, int(4 * intensity))
            cv2.circle(overlay, palm_center, ring_radius, ring_color, thickness)

        # Draw bright core
        core_radius = int(base_radius * 0.4 * pulse)
        cv2.circle(overlay, palm_center, core_radius, core_color, -1)

        # Add radiating lines (energy discharge effect)
        num_rays = 8
        ray_length = int(base_radius * 1.8)
        for i in range(num_rays):
            angle = (i / num_rays) * 2 * np.pi + time * 2
            end_x = int(palm_center[0] + np.cos(angle) * ray_length)
            end_y = int(palm_center[1] + np.sin(angle) * ray_length)

            # Pulsing ray brightness
            ray_alpha = intensity * pulse * 0.3
            cv2.line(overlay, palm_center, (end_x, end_y), ring_color, 2)

        # Blend with strong additive effect for glow
        alpha = min(0.8, intensity * 1.2)
        output = cv2.addWeighted(frame, 1, overlay, alpha, 0)

        # Add extra bloom effect
        glow_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.circle(glow_mask, palm_center, base_radius * 2, 255, -1)
        glow_mask = cv2.GaussianBlur(glow_mask, (51, 51), 0)
        glow_mask = glow_mask[:, :, np.newaxis] / 255.0

        bloom_color = np.array([255, 180, 80], dtype=np.float32)  # Cyan bloom
        bloom = (glow_mask * bloom_color * intensity * 0.5).astype(np.uint8)
        output = cv2.add(output, bloom)

        return output
