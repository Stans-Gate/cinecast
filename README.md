# 🎬 CineCast - Cinematic Gesture Control System

Single-hand cinematic effects controlled by gestures. Lock into a mode and control intensity with palm openness!

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
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
├── effects/                   # 🎨 All visual effects (teammates work here!)
│   ├── __init__.py           # Register effects here
│   ├── base_effect.py        # Base class for all effects
│   ├── zoom_effect.py        # 👍 Dolly Zoom
│   ├── rotate_effect.py      # ✌️ Rotate
│   ├── blur_effect.py        # 🤘 Motion Blur
│   └── filter_effect.py      # 👌 Color Grade
│
└── cinecast.py               # Old monolithic file (can be deleted)
```

## 👥 Team Collaboration Guide

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

1. **Start**: NO MODE (camera passthrough)
2. **Lock**: Make a gesture (hold 0.5s) → locks into that mode
3. **Control**: Open/close palm → adjusts intensity (0-100%)
4. **Quit**: Middle finger up → returns to NO MODE

### Available Gestures

| Gesture | Effect | Description |
|---------|--------|-------------|
| 👍 Thumbs Up | Dolly Zoom | Cinematic zoom in/out |
| ✌️ Peace Sign | Rotate | Continuous rotation |
| 🤘 Rock Sign | Motion Blur | Variable blur intensity |
| 👌 OK Sign | Color Grade | Cinematic color grading |
| 🖕 Middle Finger | **QUIT** | Exit current mode |

## 🔧 Configuration

Edit `main.py` to tune parameters:

```python
GESTURE_STABILITY_FRAMES = 8   # Hold gesture for ~0.5s
QUIT_GESTURE_FRAMES = 10       # Quit gesture stability
INTENSITY_SMOOTHING = 0.15     # Lower = smoother intensity
MODE_TRANSITION_SPEED = 0.2    # Crossfade speed between modes
```

## 🎨 Effect Development Tips

### Example Effects You Can Build:

- **Vignette**: Darken edges based on intensity
- **Chromatic Aberration**: RGB channel shift
- **Film Grain**: Add noise texture
- **Lens Distortion**: Fisheye or barrel distortion
- **Pixelate**: Retro pixel effect
- **Edge Detection**: Artistic outlines
- **Time Freeze**: Capture frame and overlay
- **Split Screen**: Multiple simultaneous effects

### Useful OpenCV Functions:

```python
# Color manipulation
cv2.cvtColor()        # Convert color spaces
cv2.applyColorMap()   # Apply color lookup tables

# Transformations
cv2.warpAffine()      # Rotate, scale, shear
cv2.warpPerspective() # 3D perspective transforms

# Filters
cv2.GaussianBlur()    # Blur
cv2.bilateralFilter() # Edge-preserving blur
cv2.Canny()           # Edge detection

# Blending
cv2.addWeighted()     # Alpha blend two images
```

## 📝 Notes for Teammates

- **Each effect is independent** - no conflicts!
- **Test individually** - your effect won't break others
- **Use `intensity`** - maps directly to palm openness
- **Use `time`** - for animated/oscillating effects
- **Return BGR format** - OpenCV's default color space
- **Unique `mode_id`** - increment from the last one

## 🐛 Troubleshooting

**Effect not showing up?**
- Check that it's imported in `effects/__init__.py`
- Check that it's added to `AVAILABLE_EFFECTS` list

**Gesture not detecting?**
- Increase `GESTURE_STABILITY_FRAMES` for more stability
- Check `gesture_recognition.py` for gesture mapping

**Performance issues?**
- Use `intensity < 0.05` check to skip processing
- Reduce frame resolution in `main.py`

## 📄 License

MIT - Feel free to use and modify!

---

**Happy coding! 🎬✨**
