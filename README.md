# 🎬 CineCast - Cinematic Gesture Control System

Single-hand cinematic effects controlled by gestures. Lock into a mode and control intensity with palm openness!

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python main.py
```

## 📁 Project Structure

```
cinecast/
├── main.py                    # Main application (orchestrates everything)
├── gesture_recognition.py     # Hand gesture detection logic
├── ui_renderer.py             # UI drawing functions
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── effects/                   # 🎨 All visual effects
   ├── __init__.py           # Register effects here
   ├── base_effect.py        # Base class for all effects
   ├── zoom_effect.py        # 👍 Zoom
   ├── rotate_effect.py      # ✌️ Rotate
   ├── blur_effect.py        # 🤘 Motion Blur
   ├── filter_effect.py      # 👌 Color Grade
   └── object_3d_effect.py   # 🎲 3D Object Interaction
```


### **Adding a New Effect** (for teammates)

Each teammate can work on their own effect independently!

#### 1. Create a new file in `effects/`

```python
# effects/my_new_effect.py

from effects.base_effect import BaseEffect
import cv2
import numpy as np

class MyNewEffect(BaseEffect):
    def __init__(self):
        super().__init__(
            name="My Cool Effect",  # Display name
            icon="🔥",               # Emoji icon
            mode_id=5                # Unique ID (increment from last)
        )

    def apply(self, frame, intensity, time):
        """
        Apply your effect here!

        Args:
            frame: Input frame (BGR, numpy array)
            intensity: 0.0 to 1.0 (controlled by palm openness)
            time: Current time in seconds (for animations)

        Returns:
            Modified frame (BGR, numpy array)
        """
        # Your effect code here!
        output = frame.copy()
        # ... do something cool ...
        return output
```

#### 2. Register your effect in `effects/__init__.py`

```python
from effects.my_new_effect import MyNewEffect

AVAILABLE_EFFECTS = [
    ZoomEffect(),
    RotateEffect(),
    BlurEffect(),
    FilterEffect(),
    MyNewEffect(),  # Add your effect here!
]
```

#### 3. Assign a gesture (optional)

Edit `gesture_recognition.py` → `classify_mode_gesture()` to map a gesture to your `mode_id`.

#### 4. Test it!

```bash
python main.py
```

Your effect will now appear in the app!

## 🎮 How It Works

1. **Start**: NO MODE (camera passthrough) - Menu visible on the right
2. **Navigate**: Index finger swipe up/down → scroll through menu
3. **Select**: OK sign (thumb + index touching) → lock into selected mode
4. **Special**: Raise both hands with open palms → trigger "Absolute Cinema" effect
5. **Control**: Open/close palm → adjusts intensity (0-100%) or scale (3D mode)
6. **Quit**: Thumbs down → returns to NO MODE (when locked)

### Menu System

When **UNLOCKED** (menu visible):

-   **Index Finger Swipe Up/Down**: Scroll through available modes
-   **OK Sign**: Select highlighted mode (thumb and index finger touching)
-   Menu shows all available effects

### Gesture Details

-   **Menu Scrolling**: Uses index finger tip movement for precise control
-   **Menu Selection**: OK sign (thumb touching index finger, other fingers extended)
-   **Absolute Cinema**: Raise both hands with open palms (all fingers extended)
-   **Quit**: Thumbs down gesture (thumb extended downward, other fingers closed)
-   **3D Mode**: Move hand to rotate, palm open/close to scale

When **LOCKED** (mode active):

-   **Palm Openness**: Controls intensity (most modes) or scale (3D mode)
-   **3D Object Mode**: Move hand to rotate, palm open/close to scale
-   **Fist**: Quit back to menu

### Available Modes

| Mode             | Icon | Description                                |
| ---------------- | ---- | ------------------------------------------ |
| Dolly Zoom       | 👍   | Cinematic zoom in/out                      |
| Rotate           | ✌️   | Continuous rotation                        |
| Motion Blur      | 🤘   | Variable blur intensity                    |
| Color Grade      | 👌   | Cinematic color grading                    |
| 3D Object        | 🎲   | Interactive 3D model with gesture controls |
| Iron Man Gauntlet| 🦾   | AR gauntlet with dynamic lighting          |
| Absolute Cinema  | 🎬   | B&W cinematic mode (TWO HANDS gesture)     |
| **QUIT**         | ✊   | Exit current mode (fist gesture)           |

AI used for project setup and initializing basic effects.