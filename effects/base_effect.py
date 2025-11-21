"""
Base Effect Class
=================
All cinematic effects inherit from this class.
Each teammate can create a new effect by:
1. Creating a new file in effects/
2. Inheriting from BaseEffect
3. Implementing apply() method
"""

from abc import ABC, abstractmethod
import cv2
import numpy as np


class BaseEffect(ABC):
    """Abstract base class for all cinematic effects"""

    def __init__(self, name: str, icon: str, mode_id: int):
        """
        Args:
            name: Display name of the effect (e.g., "Dolly Zoom")
            icon: Emoji icon for UI (e.g., "👍")
            mode_id: Unique identifier for this effect
        """
        self.name = name
        self.icon = icon
        self.mode_id = mode_id

    @abstractmethod
    def apply(self, frame: np.ndarray, intensity: float, time: float) -> np.ndarray:
        """
        Apply the cinematic effect to a frame.

        Args:
            frame: Input frame (BGR format)
            intensity: Effect intensity (0.0 = no effect, 1.0 = max effect)
            time: Current time in seconds (for animated effects)

        Returns:
            Modified frame (BGR format)
        """
        pass

    def reset(self):
        """
        Reset any internal state (called when effect is deactivated).
        Override if your effect needs to maintain state.
        """
        pass
