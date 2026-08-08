"""OpenCV-first candidate generation, filtering, scoring, and tracking for holes."""

from .config import DetectorConfig, ScoreWeights
from .detector import HoleDetector
from .models import Hole

__all__ = [
    "DetectorConfig",
    "ScoreWeights",
    "HoleDetector",
    "Hole",
]
