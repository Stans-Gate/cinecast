"""
3D Object Interaction Effect
=============================
Interactive 3D model rendering with gesture-based controls
Uses pyrender for proper 3D rendering
"""

import cv2
import numpy as np
from effects.base_effect import BaseEffect
import math

try:
    import trimesh
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("[WARNING] pyrender or trimesh not installed. 3D rendering will be disabled.")
    print("Install with: pip install trimesh pyrender")


class Object3DEffect(BaseEffect):
    """3D object interaction effect - render and control 3D models with gestures"""

    def __init__(self):
        super().__init__(name="3D Object", icon="🎲", mode_id=5)
        
        # 3D model state
        self.rotation_x = 0.0  # Rotation around X-axis (pitch)
        self.rotation_y = 0.0  # Rotation around Y-axis (yaw)
        self.rotation_z = 0.0  # Rotation around Z-axis (roll)
        self.scale = 1.0       # Scale factor
        
        # Smooth interpolation targets for jitter-free animation
        self.target_rotation_x = 0.0
        self.target_rotation_y = 0.0
        self.target_scale = 1.0
        
        # Smoothing factor (0.0 = instant, 1.0 = never updates)
        self.rotation_smoothing = 0.85  # 15% change per frame = smooth
        self.scale_smoothing = 0.90     # 10% change per frame = smooth
        
        # Initialize pyrender scene
        self.scene = None
        self.renderer = None
        self.mesh_node = None
        self.edge_mesh_node = None  # Edge outline mesh for visible edges
        self.model_loaded = False
        
        # Store previous hand angles to detect actual rotation changes
        self.previous_hand_angle_x = None
        self.previous_hand_angle_y = None
        
        # Angle smoothing buffer to filter out noise
        self.angle_buffer_x = []
        self.angle_buffer_y = []
        self.angle_buffer_size = 5  # Average over 5 frames
        
        if PYRENDER_AVAILABLE:
            self._setup_scene()

    def _setup_scene(self):
        """Initialize pyrender scene and load 3D model"""
        try:
            # Create scene with transparent background for AR overlay
            # bg_color=[0, 0, 0, 0] means fully transparent black background
            self.scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[0, 0, 0, 0])
            
            # Create renderer at higher resolution for better quality
            # Higher quality rendering for crisp edges
            self.render_width = 640
            self.render_height = 480
            self.renderer = pyrender.OffscreenRenderer(self.render_width, self.render_height)
            
            # Create an icosahedron with clearly defined edges
            try:
                # Create icosahedron (20-sided shape) - has clear edges and facets
                # If icosahedron doesn't exist, fall back to box (cube)
                if hasattr(trimesh.creation, 'icosahedron'):
                    mesh = trimesh.creation.icosahedron()
                elif hasattr(trimesh.creation, 'octahedron'):
                    mesh = trimesh.creation.octahedron()
                else:
                    # Fall back to box (cube) which should always exist
                    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
                
                # High-tech transparent material with black visible edges
                # Semi-transparent cyan-blue material with high reflectivity
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.95,
                    roughnessFactor=0.0,  # Very reflective/shiny for high-tech look
                    baseColorFactor=[0.15, 0.7, 1.0, 0.65]  # Bright cyan-blue with transparency (alpha=0.65)
                )
                
                # Create pyrender mesh from trimesh with smooth shading for clear edges
                pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
                
                # Add edge mesh for black visible edges
                # Create a slightly larger mesh with black material for edge outline
                try:
                    # Scale up mesh slightly for edge outline effect
                    edge_mesh_geom = mesh.copy()
                    edge_scale = 1.015  # 1.5% larger for subtle edge outline
                    edge_mesh_geom.apply_scale(edge_scale)
                    
                    # Black edge material - fully opaque for visibility
                    edge_material = pyrender.MetallicRoughnessMaterial(
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                        baseColorFactor=[0.0, 0.0, 0.0, 1.0]  # Solid black edges
                    )
                    
                    # Create edge mesh
                    edge_mesh = pyrender.Mesh.from_trimesh(edge_mesh_geom, material=edge_material, smooth=False)
                    self.edge_mesh_node = self.scene.add(edge_mesh)
                except Exception as e:
                    print(f"[3D] Could not create edge mesh: {e}")
                    self.edge_mesh_node = None
                
                # For high-tech look, we'll rely on lighting and material properties
                # rather than wireframe overlay
                self.wireframe_node = None
                
                # Add to scene
                self.mesh_node = self.scene.add(pyrender_mesh)
                
                # Add lights for high-tech look with strong edge definition
                # Main key light - bright and focused
                key_light = pyrender.DirectionalLight(color=[0.9, 0.95, 1.0], intensity=4.0)
                key_light_pose = np.eye(4)
                key_light_pose[:3, 3] = [3, 4, 5]
                self.scene.add(key_light, pose=key_light_pose)
                
                # Fill light from opposite side
                fill_light = pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=2.0)
                fill_light_pose = np.eye(4)
                fill_light_pose[:3, 3] = [-3, 2, 4]
                self.scene.add(fill_light, pose=fill_light_pose)
                
                # Rim light for edge definition - bright cyan for high-tech glow
                rim_light = pyrender.DirectionalLight(color=[0.5, 0.9, 1.0], intensity=3.5)
                rim_light_pose = np.eye(4)
                rim_light_pose[:3, 3] = [0, -3, 3]
                self.scene.add(rim_light, pose=rim_light_pose)
                
                # Additional point light for extra glow effect
                point_light = pyrender.PointLight(color=[0.6, 0.85, 1.0], intensity=2.5)
                point_light_pose = np.eye(4)
                point_light_pose[:3, 3] = [0, 0, 3]
                self.scene.add(point_light, pose=point_light_pose)
                
                self.model_loaded = True
                print("[3D] Model loaded successfully")
                
            except Exception as e:
                print(f"[3D] Error loading model: {e}")
                import traceback
                traceback.print_exc()
                self.model_loaded = False
                
        except Exception as e:
            print(f"[3D] Error setting up scene: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = False
        else:
            # If we got here without exception, model should be loaded
            if not self.model_loaded:
                print("[3D] WARNING: Scene setup completed but model_loaded is False")

    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Render 3D object on frame
        intensity: 0.0 = hidden, 1.0 = fully visible (currently not used, scale controlled separately)
        """
        if not PYRENDER_AVAILABLE:
            # Fallback: draw a simple message
            h, w = frame.shape[:2]
            cv2.putText(frame, "PyRender library not installed", (w//2 - 180, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Install: pip install pyrender trimesh", (w//2 - 220, h//2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            return frame
        
        if not self.model_loaded:
            # Try to reload the model
            print("[3D] Attempting to reload model...")
            if PYRENDER_AVAILABLE:
                self._setup_scene()
            if not self.model_loaded:
                h, w = frame.shape[:2]
                cv2.putText(frame, "3D Model failed to load", (w//2 - 140, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Check console for error messages", (w//2 - 200, h//2 + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                return frame
        
        h, w = frame.shape[:2]
        
        try:
            # Use fixed camera position - always looking at center
            # Fixed camera distance - don't adjust based on scale
            camera_distance = 5.0  # Fixed distance, scale doesn't affect camera
            
            # Fixed camera pose - always looking at origin from front
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = [0, 0, camera_distance]  # Position camera in front
            camera_pose[:3, :3] = np.array([  # Look at origin
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
            
            # Create rotation matrices for object rotation (not camera)
            # Yaw rotation (around Y axis)
            yaw_matrix = np.array([
                [math.cos(self.rotation_y), 0, math.sin(self.rotation_y), 0],
                [0, 1, 0, 0],
                [-math.sin(self.rotation_y), 0, math.cos(self.rotation_y), 0],
                [0, 0, 0, 1]
            ])
            
            # Pitch rotation (around X axis)
            pitch_matrix = np.array([
                [1, 0, 0, 0],
                [0, math.cos(self.rotation_x), -math.sin(self.rotation_x), 0],
                [0, math.sin(self.rotation_x), math.cos(self.rotation_x), 0],
                [0, 0, 0, 1]
            ])
            
            # Combine rotations - rotate object, not camera
            rotation_matrix = yaw_matrix @ pitch_matrix
            
            # Update mesh node transform (rotation, scale, and center)
            # Build transform: scale first, then rotate, keep at origin
            mesh_transform = np.eye(4)
            # Apply rotation
            mesh_transform[:3, :3] = rotation_matrix[:3, :3]
            # Apply scale (fixed size, not affected by camera distance)
            scale_matrix = np.eye(4)
            scale_matrix[:3, :3] *= self.scale * 0.8  # Scale factor
            # Combine: rotation then scale
            mesh_transform = mesh_transform @ scale_matrix
            mesh_transform[:3, 3] = [0, 0, 0]  # Keep object centered at origin
            
            if self.mesh_node is not None:
                self.scene.set_pose(self.mesh_node, mesh_transform)
            
            # Update edge mesh transform to match main mesh
            if self.edge_mesh_node is not None:
                self.scene.set_pose(self.edge_mesh_node, mesh_transform)
            
            # Create camera - fixed position, fixed view (always looking at origin)
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=self.render_width / self.render_height)
            camera_node = self.scene.add(camera, pose=camera_pose)
            
            # Render with alpha channel support
            # Use flags to ensure proper alpha rendering
            flags = pyrender.RenderFlags.RGBA
            color, depth = self.renderer.render(self.scene, flags=flags)
            
            # Remove camera node for next frame
            self.scene.remove_node(camera_node)
            
            # Check color format and extract channels
            render_h, render_w = color.shape[:2]
            
            # Pyrender should output RGBA with transparent background
            # Extract alpha channel for proper AR overlay
            if color.shape[2] == 4:
                # RGBA image - extract alpha channel properly
                # Extract alpha as 2D array first, then expand to 3D
                alpha_2d = color[:, :, 3].astype(np.float32) / 255.0  # Normalize alpha to 0-1
                alpha = np.expand_dims(alpha_2d, axis=2)  # Shape: (H, W, 1)
                color_rgb = color[:, :, :3]  # Extract RGB channels
            elif color.shape[2] == 3:
                # RGB image - create alpha from depth (white pixels = object, black = transparent)
                # Use depth buffer to create alpha mask
                depth_normalized = (depth > 0).astype(np.float32)
                alpha = np.expand_dims(depth_normalized, axis=2)
                color_rgb = color
            else:
                # Unexpected format - use full opacity
                print(f"[3D] Unexpected color format: {color.shape}")
                alpha = np.ones((render_h, render_w, 1), dtype=np.float32)
                color_rgb = color[:, :, :3] if color.shape[2] >= 3 else color
            
            # Convert RGB to BGR for OpenCV
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            
            # Upscale from render resolution to frame resolution
            if render_h != h or render_w != w:
                color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
                # Resize alpha while preserving single channel
                alpha_resized = cv2.resize(alpha.squeeze(), (w, h), interpolation=cv2.INTER_LINEAR)
                alpha = np.expand_dims(alpha_resized, axis=2)
            
            # Ensure alpha has correct shape for broadcasting (H, W, 1)
            if len(alpha.shape) == 2:
                alpha = np.expand_dims(alpha, axis=2)
            
            # Composite 3D render onto frame using alpha blending (AR-style overlay)
            frame_float = frame.astype(np.float32)
            color_float = color_bgr.astype(np.float32)
            
            # Ensure dimensions match for broadcasting
            if frame_float.shape != color_float.shape:
                color_float = cv2.resize(color_float, (frame_float.shape[1], frame_float.shape[0]))
            if alpha.shape[:2] != frame_float.shape[:2]:
                alpha = cv2.resize(alpha.squeeze(), (frame_float.shape[1], frame_float.shape[0]), interpolation=cv2.INTER_LINEAR)
                if len(alpha.shape) == 2:
                    alpha = np.expand_dims(alpha, axis=2)
            
            # Alpha blend: result = background * (1 - alpha) + foreground * alpha
            # This creates AR-style overlay where transparent areas show camera feed
            blended = frame_float * (1 - alpha) + color_float * alpha
            
            output = blended.astype(np.uint8)
            
            return output
            
        except Exception as e:
            print(f"[3D] Error rendering: {e}")
            return frame

    def update_rotation(self, hand_angle_x, hand_angle_y):
        """
        Update rotation to directly match hand orientation
        hand_angle_x, hand_angle_y: Direct hand rotation angles in radians
        Only updates when there's a significant change to prevent jitter
        """
        # Only update if hand angles are provided (not None)
        if hand_angle_x is not None and hand_angle_y is not None:
            # Add to smoothing buffer
            self.angle_buffer_x.append(hand_angle_x)
            self.angle_buffer_y.append(hand_angle_y)
            
            # Keep buffer size limited
            if len(self.angle_buffer_x) > self.angle_buffer_size:
                self.angle_buffer_x.pop(0)
                self.angle_buffer_y.pop(0)
            
            # Average angles over buffer to filter noise
            if len(self.angle_buffer_x) >= 3:  # Need at least 3 samples
                smoothed_angle_x = sum(self.angle_buffer_x) / len(self.angle_buffer_x)
                smoothed_angle_y = sum(self.angle_buffer_y) / len(self.angle_buffer_y)
                
                # Only update if change is significant (threshold to prevent jitter)
                angle_change_threshold = 0.05  # ~3 degrees in radians
                
                if self.previous_hand_angle_x is not None and self.previous_hand_angle_y is not None:
                    # Check if change is significant
                    change_x = abs(smoothed_angle_x - self.previous_hand_angle_x)
                    change_y = abs(smoothed_angle_y - self.previous_hand_angle_y)
                    
                    # Only update if change exceeds threshold
                    if change_x > angle_change_threshold or change_y > angle_change_threshold:
                        self.target_rotation_x = smoothed_angle_x
                        self.target_rotation_y = smoothed_angle_y
                        # Update previous angles
                        self.previous_hand_angle_x = smoothed_angle_x
                        self.previous_hand_angle_y = smoothed_angle_y
                else:
                    # First time - set directly
                    self.target_rotation_x = smoothed_angle_x
                    self.target_rotation_y = smoothed_angle_y
                    self.previous_hand_angle_x = smoothed_angle_x
                    self.previous_hand_angle_y = smoothed_angle_y
        
        # Apply smoothing interpolation to actual rotation values
        smoothing = self.rotation_smoothing
        self.rotation_x += (self.target_rotation_x - self.rotation_x) * (1 - smoothing)
        self.rotation_y += (self.target_rotation_y - self.rotation_y) * (1 - smoothing)
        
        # Normalize angles to prevent overflow
        self.rotation_x = self.rotation_x % (2 * math.pi)
        self.rotation_y = self.rotation_y % (2 * math.pi)
        self.target_rotation_x = self.target_rotation_x % (2 * math.pi)
        self.target_rotation_y = self.target_rotation_y % (2 * math.pi)

    def update_scale(self, new_scale):
        """Update scale based on palm openness - smoothed"""
        # Clamp target scale to reasonable range
        self.target_scale = np.clip(new_scale, 0.5, 2.5)
        
        # Apply smoothing interpolation to actual scale value
        smoothing = self.scale_smoothing
        self.scale += (self.target_scale - self.scale) * (1 - smoothing)

    def reset(self):
        """Reset 3D object state"""
        self.rotation_x = 0.0
        self.rotation_y = 0.0
        self.rotation_z = 0.0
        self.scale = 1.0
        self.target_rotation_x = 0.0
        self.target_rotation_y = 0.0
        self.target_scale = 1.0
        self.previous_hand_angle_x = None
        self.previous_hand_angle_y = None
        self.angle_buffer_x = []
        self.angle_buffer_y = []
