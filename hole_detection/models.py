from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Hole:
    """A scored opening candidate, ready for a mapper or navigation planner."""

    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area_px: float
    aspect_ratio: float
    solidity: float
    local_contrast: float
    boundary_edge_fraction: float
    area_score: float
    temporal_stability: float
    score: float
    diameter_cm: Optional[float]
    robot_fits: Optional[bool]
    track_id: Optional[int] = None
    confirmed: bool = False
