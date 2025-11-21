"""
Effects Package
===============
Import all available effects here.
To add a new effect:
1. Create new_effect.py in this folder
2. Import it here
3. Add to AVAILABLE_EFFECTS list
"""

from effects.zoom_effect import ZoomEffect
from effects.rotate_effect import RotateEffect
from effects.blur_effect import BlurEffect
from effects.filter_effect import FilterEffect

# List of all available effects
# Add new effects here to make them available in the app
AVAILABLE_EFFECTS = [
    ZoomEffect(),
    RotateEffect(),
    BlurEffect(),
    FilterEffect(),
]

__all__ = ['AVAILABLE_EFFECTS', 'ZoomEffect', 'RotateEffect', 'BlurEffect', 'FilterEffect']
