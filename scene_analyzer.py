"""
Scene Analyzer Module
=====================
Detects scene characteristics and suggests appropriate VFX effects.

Features:
- Brightness analysis (dark scenes)
- MediaPipe object detection (books, cups, laptops, etc.)
- Contextual VFX suggestions
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class SceneAnalyzer:
    """Analyzes scenes and provides VFX suggestions using MediaPipe"""

    def __init__(self, model_path='models/efficientdet_lite0.tflite'):
        self.suggestion_cooldown = 2.0  # Seconds between suggestion updates
        self.last_suggestion_time = 0
        self.current_suggestion = None
        self.confidence_threshold = 0.4  # Lower threshold for MediaPipe (more sensitive)

        # Detection thresholds
        self.dark_threshold = 60  # Average brightness below this = dark scene
        self.bright_threshold = 180  # Average brightness above this = bright scene

        # Initialize MediaPipe Object Detector
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                max_results=5,  # Detect up to 5 objects
                score_threshold=0.3,  # Minimum confidence for detection
                category_allowlist=None  # Allow all categories
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
            print("[SCENE] MediaPipe Object Detector initialized")
        except Exception as e:
            print(f"[WARNING] Failed to initialize Object Detector: {e}")
            self.detector = None

        # Object categories that map to VFX suggestions
        # MediaPipe uses COCO dataset labels
        self.object_to_effect_map = {
            'book': ('Zoom', 'Book detected - perfect for zoom'),
            # 'bottle': ('Rotate', 'Bottle detected - try rotation'),
            # 'laptop': ('Zoom', 'Laptop detected - focus with zoom'),
            # 'keyboard': ('Zoom', 'Keyboard detected - focus with zoom'),
            # 'cell phone': ('Zoom', 'Phone detected - focus with zoom'),
            'mouse': ('Zoom', 'Mouse detected - focus with zoom'),
        }

    def analyze_frame(self, frame, current_time):
        """
        Analyze frame and return VFX suggestion if relevant

        Args:
            frame: Input frame (BGR)
            current_time: Current time in seconds

        Returns:
            dict with suggestion info or None if no suggestion
            {
                'effect_name': str,
                'reason': str,
                'confidence': float (0-1),
                'highlight_region': tuple or None (x, y, w, h)
            }
        """
        # Only update suggestions periodically to avoid flickering
        if current_time - self.last_suggestion_time < self.suggestion_cooldown:
            return self.current_suggestion

        # Analyze scene characteristics
        brightness_level = self._analyze_brightness(frame)
        detected_objects = self._detect_objects(frame)

        # Determine suggestion based on analysis
        suggestion = None

        # Priority 1: Object detection (more specific)
        if detected_objects:
            suggestion = self._suggest_for_objects(detected_objects)

        # Priority 2: Brightness analysis (general)
        if not suggestion:
            suggestion = self._suggest_for_brightness(brightness_level)

        # Update state
        if suggestion and suggestion['confidence'] >= self.confidence_threshold:
            self.current_suggestion = suggestion
            self.last_suggestion_time = current_time
        else:
            # Clear suggestion if confidence too low
            self.current_suggestion = None

        return self.current_suggestion

    def _analyze_brightness(self, frame):
        """
        Analyze overall brightness of the frame

        Returns:
            float: Average brightness (0-255)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate average brightness
        avg_brightness = np.mean(gray)

        return avg_brightness

    def _detect_objects(self, frame):
        """
        Detect objects in the frame using MediaPipe Object Detector

        Returns:
            list of detected objects with properties
        """
        if self.detector is None:
            return []

        detected_objects = []

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect objects
            detection_result = self.detector.detect(mp_image)

            # Process detections
            if detection_result.detections:
                h, w = frame.shape[:2]

                for detection in detection_result.detections:
                    # Get bounding box
                    bbox = detection.bounding_box
                    x = bbox.origin_x
                    y = bbox.origin_y
                    width = bbox.width
                    height = bbox.height

                    # Get category and confidence
                    category = detection.categories[0]
                    category_name = category.category_name.lower()
                    confidence = category.score

                    # Only include objects we have effect mappings for
                    if category_name in self.object_to_effect_map:
                        detected_objects.append({
                            'type': 'object',
                            'category': category_name,
                            'confidence': confidence,
                            'bbox': (x, y, width, height)
                        })

                # Sort by confidence
                detected_objects.sort(key=lambda x: x['confidence'], reverse=True)

        except Exception as e:
            print(f"[WARNING] Object detection failed: {e}")
            return []

        return detected_objects

    def _suggest_for_objects(self, detected_objects):
        """
        Generate VFX suggestions based on detected objects

        Returns:
            suggestion dict or None
        """
        if not detected_objects:
            return None

        # Get the most confident detection
        obj = detected_objects[0]
        category = obj['category']

        # Get effect mapping
        if category in self.object_to_effect_map:
            effect_name, reason = self.object_to_effect_map[category]

            return {
                'effect_name': effect_name,
                'reason': reason,
                'confidence': obj['confidence'],
                'highlight_region': obj['bbox']
            }

        return None

    def _suggest_for_brightness(self, brightness_level):
        """
        Generate VFX suggestions based on brightness

        Returns:
            suggestion dict or None
        """
        if brightness_level < self.dark_threshold:
            # Dark scene - suggest Iron Man effect for illumination
            darkness_level = 1.0 - (brightness_level / self.dark_threshold)
            return {
                'effect_name': 'Iron Man Gauntlet',
                'reason': 'Dark scene detected',
                'confidence': min(1.0, darkness_level * 1.2),
                'highlight_region': None
            }

        elif brightness_level > self.bright_threshold:
            # Very bright scene - suggest color grading
            brightness_excess = (brightness_level - self.bright_threshold) / (255 - self.bright_threshold)
            return {
                'effect_name': 'Color Grade',
                'reason': 'Bright scene detected',
                'confidence': min(1.0, brightness_excess * 1.2),
                'highlight_region': None
            }

        # Normal lighting - no suggestion
        return None

    def reset(self):
        """Reset analyzer state"""
        self.current_suggestion = None
        self.last_suggestion_time = 0
