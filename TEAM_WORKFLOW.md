# 👥 Team Workflow Guide

## 📂 New Project Structure

```
cinecast/
│
├── 🚀 main.py                    ← Main application (don't modify often)
├── 🎨 gesture_recognition.py     ← Hand gesture logic
├── 🖼️  ui_renderer.py             ← UI drawing
├── 📋 requirements.txt
├── 📖 README.md
│
├── 🎬 effects/                    ← **TEAMMATES WORK HERE!**
│   ├── __init__.py               ← Register your effect here
│   ├── base_effect.py            ← Base class (inherit from this)
│   ├── TEMPLATE_new_effect.py    ← Copy this to create new effects
│   │
│   ├── zoom_effect.py            ← Example: Teammate 1
│   ├── rotate_effect.py          ← Example: Teammate 2
│   ├── blur_effect.py            ← Example: Teammate 3
│   └── filter_effect.py          ← Example: Teammate 4
│
└── cinecast.py                   ← Old file (can delete after testing)
```

---

## 🎯 Quick Start for New Teammates

### Step 1: Clone and Test

```bash
cd cinecast
python main.py  # Make sure it works first!
```

### Step 2: Create Your Effect

```bash
# Copy the template
cp effects/TEMPLATE_new_effect.py effects/my_effect.py

# Edit your file
# (Use your favorite editor)
```

### Step 3: Implement Your Effect

Open `effects/my_effect.py`:

```python
from effects.base_effect import BaseEffect
import cv2
import numpy as np

class MyEffect(BaseEffect):
    def __init__(self):
        super().__init__(
            name="My Awesome Effect",
            icon="✨",
            mode_id=5  # Next available ID
        )

    def apply(self, frame, intensity, time):
        # Your magic here!
        output = frame.copy()

        # Example: Add a color tint
        tint = np.full_like(frame, (255, 0, 0))  # Blue tint
        output = cv2.addWeighted(output, 1 - intensity, tint, intensity, 0)

        return output
```

### Step 4: Register Your Effect

Edit `effects/__init__.py`:

```python
from effects.my_effect import MyEffect  # Add this import

AVAILABLE_EFFECTS = [
    ZoomEffect(),
    RotateEffect(),
    BlurEffect(),
    FilterEffect(),
    MyEffect(),  # Add your effect here!
]
```

### Step 5: Test!

```bash
python main.py
```

Your effect is now live! 🎉

---

## 🔄 Git Workflow (Multiple Teammates)

### Working on Different Effects

Since each effect is in its own file, you won't have merge conflicts!

```bash
# Teammate 1: Working on vignette
git checkout -b feature/vignette-effect
# Edit effects/vignette_effect.py
git add effects/vignette_effect.py effects/__init__.py
git commit -m "Add vignette effect"
git push origin feature/vignette-effect

# Teammate 2: Working on film grain (at the same time!)
git checkout -b feature/film-grain
# Edit effects/film_grain_effect.py
git add effects/film_grain_effect.py effects/__init__.py
git commit -m "Add film grain effect"
git push origin feature/film-grain
```

### Only Merge Conflict: `effects/__init__.py`

If two teammates add effects at the same time, you'll need to merge the `AVAILABLE_EFFECTS` list:

```python
# Both teammates added their effect, resolve like this:
AVAILABLE_EFFECTS = [
    ZoomEffect(),
    RotateEffect(),
    BlurEffect(),
    FilterEffect(),
    VignetteEffect(),    # Teammate 1's effect
    FilmGrainEffect(),   # Teammate 2's effect
]
```

---

## 🎨 Effect Ideas by Difficulty

### 🟢 Easy (Good for beginners)

1. **Brightness/Contrast**
   ```python
   output = cv2.convertScaleAbs(frame, alpha=1 + intensity, beta=intensity * 50)
   ```

2. **Color Tint**
   ```python
   tint = np.full_like(frame, (B, G, R))
   output = cv2.addWeighted(frame, 1 - intensity, tint, intensity, 0)
   ```

3. **Invert Colors**
   ```python
   output = cv2.bitwise_not(frame)
   output = cv2.addWeighted(frame, 1 - intensity, output, intensity, 0)
   ```

### 🟡 Medium

1. **Vignette** (darken edges)
2. **Film Grain** (add noise)
3. **Pixelate** (reduce resolution)
4. **Edge Detection** (Canny edges)
5. **Sepia Tone** (vintage look)

### 🔴 Advanced

1. **Chromatic Aberration** (RGB shift)
2. **Lens Distortion** (fisheye effect)
3. **Kaleidoscope** (symmetrical patterns)
4. **Glitch Effect** (digital artifacts)
5. **Time Freeze** (capture and overlay frames)

---

## 📝 Code Style Guidelines

### 1. File Naming
- Use `snake_case`: `my_effect.py`
- End with `_effect.py`

### 2. Class Naming
- Use `PascalCase`: `MyEffect`
- End with `Effect`

### 3. Documentation
- Add docstring to your class
- Explain what your effect does
- Mention any special parameters

### 4. Performance
- Check `if intensity < 0.05: return frame` to skip processing
- Avoid heavy operations in the main loop
- Test with webcam to ensure smooth FPS

---

## 🐛 Debugging Tips

### Effect Not Showing Up?

1. Check import in `effects/__init__.py`
2. Check it's in `AVAILABLE_EFFECTS` list
3. Check for syntax errors: `python -m py_compile effects/my_effect.py`

### Effect Crashes the App?

1. Add try-except:
   ```python
   def apply(self, frame, intensity, time):
       try:
           # Your code
           return output
       except Exception as e:
           print(f"Error in {self.name}: {e}")
           return frame  # Return original on error
   ```

2. Check frame dimensions: `h, w = frame.shape[:2]`
3. Make sure you return BGR format (not RGB or grayscale)

### Low FPS?

1. Reduce processing: use smaller kernel sizes, fewer operations
2. Skip frames: `if int(time * 30) % 2 == 0: return cached_frame`
3. Use `cv2.INTER_NEAREST` instead of `cv2.INTER_LINEAR` for faster resizing

---

## 🎓 Learning Resources

### OpenCV Tutorials
- [Official OpenCV Python Docs](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [OpenCV Filter Functions](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html)

### Effect Inspiration
- Look at Instagram/TikTok filters
- Watch cinematography tutorials on YouTube
- Explore Photoshop blend modes

### Ask for Help
- Check existing effects in `effects/` folder
- Read `TEMPLATE_new_effect.py` for examples
- Ask teammates in chat!

---

## ✅ Testing Checklist

Before committing your effect:

- [ ] Effect loads without errors
- [ ] Intensity control works (0% to 100%)
- [ ] FPS stays above 15 FPS
- [ ] No crashes with invalid input
- [ ] Docstrings added
- [ ] Code formatted nicely
- [ ] Tested with both hands (left & right)

---

**Happy coding! 🎬✨**

Questions? Check the main [README.md](README.md) or ask your teammates!
