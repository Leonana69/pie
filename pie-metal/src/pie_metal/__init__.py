"""
pie-metal: Metal-accelerated FlashInfer replacement for macOS

A drop-in replacement for FlashInfer that uses Apple's Metal Performance Shaders
for attention operations on macOS with Apple Silicon.
"""

__version__ = "0.1.0"
__author__ = "PIE Metal Team"

# Import main API for convenience
try:
    from . import ops
    __all__ = ["ops"]
except ImportError:
    # Metal backend not available, graceful degradation
    __all__ = []