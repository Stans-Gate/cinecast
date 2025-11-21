# 🏗️ CineCast Architecture

## 🔄 System Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         MAIN.PY                             │
│                    (Orchestrator)                           │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   GESTURE    │    │   EFFECTS    │    │  UI RENDERER │
│ RECOGNITION  │    │   PACKAGE    │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Hand         │    │ BaseEffect   │    │ draw_ui()    │
│ Detection    │    │   ↑          │    │ draw_guide() │
│              │    │   │          │    │              │
│ - Palm Open  │    │   ├─ Zoom   │    └──────────────┘
│ - Fingers    │    │   ├─ Rotate │
│ - Gestures   │    │   ├─ Blur   │
│              │    │   ├─ Filter │
└──────────────┘    │   └─ [NEW]  │
                    │              │
                    │ Each effect  │
                    │ implements   │
                    │ apply()      │
                    └──────────────┘
```

---

## 📦 Module Breakdown

### 1️⃣ **main.py** (Orchestrator)
**Role**: Main application loop, coordinates all modules

**Responsibilities**:
- Initialize camera and MediaPipe
- Run main event loop (capture → process → display)
- Handle keyboard input (Q to quit, R to record)
- Manage application state (locked/unlocked, current effect)
- Smooth transitions between effects

**Key Variables**:
```python
mode_locked = False          # Is a mode currently locked?
current_effect = None        # Active effect object
smoothed_intensity = 0.5     # Smoothed intensity value
gesture_buffer = deque()     # For gesture stability
```

**Don't modify unless**: Adding new global features (e.g., multi-cam support)

---

### 2️⃣ **gesture_recognition.py** (Input Handler)
**Role**: Detect and classify hand gestures

**Functions**:

```python
count_extended_fingers(landmarks)
# Returns: (count, [thumb, index, middle, ring, pinky])
# Used by: Other gesture functions

get_palm_openness(landmarks)
# Returns: 0.0 (closed) to 1.0 (open)
# Used by: Intensity control

is_quit_gesture(landmarks)
# Returns: True if middle finger up only
# Used by: Mode unlocking

classify_mode_gesture(landmarks, effects)
# Returns: effect mode_id or None
# Used by: Mode locking
```

**Modify when**: Adding new gesture types or improving detection accuracy

---

### 3️⃣ **effects/** (Effect Library)
**Role**: All visual effects live here

#### **base_effect.py** (Abstract Base Class)

```python
class BaseEffect(ABC):
    def __init__(self, name, icon, mode_id):
        # Effect metadata

    @abstractmethod
    def apply(self, frame, intensity, time):
        # Must be implemented by subclasses
        pass

    def reset(self):
        # Optional: reset internal state
        pass
```

#### **Individual Effects** (zoom, rotate, blur, filter)

Each effect:
1. Inherits from `BaseEffect`
2. Implements `apply(frame, intensity, time)`
3. Returns modified frame

**Modify when**: Creating/updating effects (this is where teammates work!)

#### **__init__.py** (Effect Registry)

```python
AVAILABLE_EFFECTS = [
    ZoomEffect(),
    RotateEffect(),
    # Add new effects here
]
```

**Modify when**: Adding new effects to the system

---

### 4️⃣ **ui_renderer.py** (Display Layer)
**Role**: Draw all UI overlays

**Functions**:

```python
draw_ui(frame, effect_name, effect_icon, intensity, ...)
# Draws: lock status, mode name, intensity bar, hand detection

draw_gesture_guide(frame, locked, effects)
# Draws: dynamic gesture guide (changes based on lock state)
```

**Modify when**: Changing UI layout or adding new indicators

---

## 🔄 Data Flow

### Frame Processing Pipeline

```
1. CAPTURE
   camera.read() → raw frame

2. HAND DETECTION
   MediaPipe → hand landmarks

3. GESTURE RECOGNITION
   landmarks → gesture type / palm openness

4. STATE MANAGEMENT
   gesture → lock/unlock mode
   palm openness → intensity value

5. EFFECT APPLICATION
   frame + intensity → effect.apply() → output frame

6. UI OVERLAY
   output + metadata → draw_ui() → final frame

7. DISPLAY & RECORD
   final frame → cv2.imshow() / video_writer.write()
```

---

## 🎯 State Machine

```
┌─────────────────┐
│   NO MODE       │  🔓 UNLOCKED
│  (Passthrough)  │
└─────────────────┘
         │
         │ Gesture detected & held (8 frames)
         ▼
┌─────────────────┐
│  EFFECT LOCKED  │  🔒 LOCKED
│  (e.g., Zoom)   │
└─────────────────┘
         │
         │ Palm open/close → intensity control
         │
         │ Middle finger up (10 frames)
         ▼
┌─────────────────┐
│   NO MODE       │  🔓 UNLOCKED
│  (Passthrough)  │
└─────────────────┘
```

**States**:
- **UNLOCKED**: Waiting for gesture to lock a mode
- **LOCKED**: Effect active, intensity controlled by palm

**Transitions**:
- Gesture → Lock into mode
- Middle finger → Unlock to NO MODE

---

## 🧩 Modularity Benefits

### ✅ Each teammate can work independently:

| Teammate | Works On | Conflicts With |
|----------|----------|----------------|
| Alice | `zoom_effect.py` | None |
| Bob | `rotate_effect.py` | None |
| Charlie | `blur_effect.py` | None |
| Diana | `vignette_effect.py` | Only `__init__.py` (easy to merge) |

### ✅ Easy to test in isolation:

```python
# Test your effect without running the full app
from effects.zoom_effect import ZoomEffect
import cv2

effect = ZoomEffect()
frame = cv2.imread("test.jpg")
output = effect.apply(frame, intensity=0.5, time=0)
cv2.imshow("Test", output)
cv2.waitKey(0)
```

### ✅ Easy to extend:

- Want to add a new gesture? → Edit `gesture_recognition.py`
- Want to add a new UI element? → Edit `ui_renderer.py`
- Want to add a new effect? → Create `effects/new_effect.py`

---

## 🔍 Key Design Decisions

### Why separate files?
- **Git-friendly**: Fewer merge conflicts
- **Testable**: Each module can be tested independently
- **Readable**: Easier to understand and navigate

### Why BaseEffect class?
- **Consistency**: All effects have same interface
- **Polymorphism**: `main.py` doesn't need to know which effect is active
- **Extensibility**: Easy to add new effects without modifying core code

### Why gesture buffers?
- **Stability**: Prevent flickering from momentary false detections
- **Debouncing**: Ensure intentional gestures (not accidental)

### Why smooth intensity?
- **Natural feel**: Gradual transitions are more cinematic
- **Reduced jitter**: Hand tracking isn't perfect

---

## 🎓 Advanced Topics

### Adding Multi-Effect Layers

Want to apply multiple effects at once?

```python
# In main.py
active_effects = [effect1, effect2]
output = frame
for effect in active_effects:
    output = effect.apply(output, intensity, time)
```

### Adding Effect Parameters

Want effects with custom settings?

```python
class ZoomEffect(BaseEffect):
    def __init__(self, max_zoom=1.5):
        super().__init__(...)
        self.max_zoom = max_zoom

    def apply(self, frame, intensity, time):
        scale = 0.7 + intensity * (self.max_zoom - 0.7)
        # ...
```

### Adding Persistent State

Want effects to remember previous frames?

```python
class TimeFreeze Effect(BaseEffect):
    def __init__(self):
        super().__init__(...)
        self.frozen_frame = None

    def apply(self, frame, intensity, time):
        if intensity > 0.9 and self.frozen_frame is None:
            self.frozen_frame = frame.copy()
        return self.frozen_frame if self.frozen_frame else frame

    def reset(self):
        self.frozen_frame = None
```

---

## 📚 Further Reading

- **OpenCV Docs**: https://docs.opencv.org/
- **MediaPipe Hands**: https://google.github.io/mediapipe/solutions/hands
- **Python ABC Module**: https://docs.python.org/3/library/abc.html

---

**Questions?** Check [README.md](README.md) or [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md)!
