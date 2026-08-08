"""Central tuning controls for every stage of the hole-detection pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class ScoreWeights:
    """Weights in your equation: S = wc*C + we*E + wa*A + wt*T.

    All four input terms are normalised to 0..1 before these weights are used.
    The defaults sum to 1.0, keeping the final score approximately in 0..1.
    Increase a value when that signal is more trustworthy in your environment.
    """

    contrast: float = 0.40  # wc: darkness relative to the outer ring.
    edge: float = 0.25  # we: Canny edge support along the boundary.
    area: float = 0.15  # wa: comfortable size above the noise limit.
    temporal: float = 0.20  # wt: repeated observation over several frames.


@dataclass(frozen=True)
class DetectorConfig:
    """Parameters likely to require tuning on the real camera.

    Start by changing values here rather than editing detector.py. Pixel values
    depend on camera resolution and distance from the observed surface.
    """

    # STEP 1.2 - Gaussian noise reduction. Larger kernels remove more noise but may
    # erase small/far-away openings. OpenCV requires an odd value: 3, 5, 7, etc.
    gaussian_kernel: int = 5

    # STEP 1.3 - CLAHE correction for uneven lighting. Raising clip_limit adds local
    # contrast but can amplify texture/noise. grid_size controls the CLAHE tile grid.
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # STEP 1.4 - Adaptive threshold. block_size is the local neighbourhood and must
    # be odd. Larger blocks respond to broader light changes. adaptive_c shifts the
    # threshold relative to that local neighbourhood; tune these two together.
    adaptive_block_size: int = 31
    adaptive_c: float = 4.0

    # STEP 1.5 - Mask cleanup. OPEN removes small blobs, CLOSE fills small gaps.
    morphology_kernel: int = 5
    morphology_iterations: int = 1

    # STEP 2 - Hard rejection filters. A rejected candidate is never scored.
    min_area_px: float = 350.0  # 200% larger than the original 150 px^2 minimum.
    max_area_fraction: float = 0.45  # Reject shadows covering most of the frame.
    # max(long side / short side). 1.0 is square; 2.0 is twice as long as wide.
    max_aspect_ratio: float = 2.0
    # solidity = contour area / convex hull area. Lower permits irregular shapes.
    min_solidity: float = 0.65
    # C = mean outside-ring intensity - mean inside intensity, on a 0..255 image.
    min_local_contrast: float = 8.0
    # E = strong boundary-edge pixels / all boundary-band pixels, in 0..1.
    min_boundary_edge_fraction: float = 0.12
    ring_width_px: int = 5  # Outside-only ring width used to calculate C.
    boundary_width_px: int = 2  # Contour band in which Canny edges count for E.
    # Increase both Canny thresholds if texture creates edges; lower for weak rims.
    canny_low: int = 50
    canny_high: int = 150

    # PHYSICAL FIT - 5 cm robot requirement. With 10% clearance, the measured
    # opening must be at least 5.5 cm across its narrower image dimension.
    robot_diameter_cm: float = 5.0
    clearance_fraction: float = 0.10
    # Pixels per centimetre at the candidate surface. This changes with distance.
    # Leave None when unknown: robot_fits will be None rather than an unsafe guess.
    pixels_per_cm: Optional[float] = None

    # TEMPORAL SCORE T - frame-to-frame association in image coordinates.
    association_distance_px: float = 50.0  # Maximum centre movement for a match.
    max_track_misses: int = 8  # Survive brief occlusion/detection loss.
    # Require a full two seconds at 30 FPS (or four seconds at 15 FPS) before the
    # live camera is allowed to display a detection box.
    confirmation_frames: int = 60  # Consecutive frames required for confirmation.

    # SCORE - tune wc/we/wa/wt in ScoreWeights above.
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self) -> None:
        """Fail early when OpenCV would reject an invalid kernel/block value."""
        if self.gaussian_kernel < 3 or self.gaussian_kernel % 2 == 0:
            raise ValueError("gaussian_kernel must be odd and at least 3")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be odd and at least 3")
        if self.pixels_per_cm is not None and self.pixels_per_cm <= 0:
            raise ValueError("pixels_per_cm must be positive")
