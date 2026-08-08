"""OpenCV candidate generation, hard filtering, measurement, and ranking."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import DetectorConfig
from .models import Hole
from .tracker import HoleTracker


class HoleDetector:
    """Orchestrate detection while exposing stable Hole objects to other modules.

    Navigation, mapping, and a future YOLO verifier can use the Hole interface
    without depending on these OpenCV implementation details.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        # Tune the pipeline through DetectorConfig instead of changing this class.
        self.config = config or DetectorConfig()
        self.tracker = HoleTracker(self.config)

    def detect(self, frame_bgr: np.ndarray, pixels_per_cm: Optional[float] = None) -> List[Hole]:
        """Run the full pipeline and return highest-score candidates first.

        A per-frame pixels_per_cm value from depth/stereo overrides fixed config.
        """
        # STEP 1.1-1.5: produce an illumination-normalised image and dark-region mask.
        gray, mask = self.generate_candidates(frame_bgr)

        # STEP 2 / E: calculate Canny once and reuse it for every candidate.
        edges = cv2.Canny(gray, self.config.canny_low, self.config.canny_high)

        # STEP 1.6: RETR_EXTERNAL gets only outer contours. Use RETR_TREE here if a
        # visible rim and its nested inner opening must be represented separately.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Live scale is preferred because perspective changes pixel size with range.
        scale = pixels_per_cm if pixels_per_cm is not None else self.config.pixels_per_cm

        # STEP 2: _measure returns None for each candidate failing a hard filter.
        holes = [hole for contour in contours if (hole := self._measure(contour, gray, edges, scale))]

        # T must be determined before S because temporal stability contributes to S.
        holes = self.tracker.update(holes)
        for hole in holes:
            hole.score = self._score(hole)

        # A planner receives the most promising candidate first.
        return sorted(holes, key=lambda hole: hole.score, reverse=True)

    def generate_candidates(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform steps 1.1-1.5 and return (processed grayscale, binary mask).

        Step 1.6 is kept in detect() so this mask can be displayed during tuning.
        """
        # STEP 1.1: discard colour for this classical detector. Grayscale arrays are
        # also accepted directly, which is useful for tests and monochrome cameras.
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr.copy()

        # STEP 1.2: suppress sensor/camera noise before enhancing local contrast.
        # Sigma=0 tells OpenCV to derive sigma from gaussian_kernel.
        blurred = cv2.GaussianBlur(gray, (self.config.gaussian_kernel, self.config.gaussian_kernel), 0)

        # STEP 1.3: CLAHE works tile-by-tile, helping with uneven lighting without
        # applying one aggressive global contrast transform to the whole frame.
        clahe = cv2.createCLAHE(self.config.clahe_clip_limit, self.config.clahe_grid_size)
        normalised = clahe.apply(blurred)

        # STEP 1.4: THRESH_BINARY_INV makes locally DARK pixels white (255) in the
        # mask, allowing findContours to treat dark areas as foreground objects.
        mask = cv2.adaptiveThreshold(
            normalised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )

        # STEP 1.5: an ellipse suits roughly round openings. OPEN removes isolated
        # white noise; CLOSE reconnects nearby white pixels and fills small gaps.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morphology_kernel, self.config.morphology_kernel),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.config.morphology_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.config.morphology_iterations)
        return normalised, mask

    def _measure(
        self,
        contour: np.ndarray,
        gray: np.ndarray,
        edges: np.ndarray,
        scale: Optional[float],
    ) -> Optional[Hole]:
        """Measure one contour, reject weak candidates, and create a Hole."""
        # BASIC FEATURES: AREA, WIDTH, HEIGHT, ASPECT RATIO, and SOLIDITY.
        area = float(cv2.contourArea(contour))
        image_area = gray.shape[0] * gray.shape[1]
        x, y, width, height = cv2.boundingRect(contour)
        # Long/short makes aspect >= 1 regardless of portrait/landscape direction.
        aspect = max(width, height) / max(1, min(width, height))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area else 0.0

        # SIZE FILTER: reject tiny noise and implausibly frame-filling shadows.
        if area < self.config.min_area_px or area > image_area * self.config.max_area_fraction:
            return None
        # SHAPE FILTER: reject extremely elongated or irregular/concave regions.
        if aspect > self.config.max_aspect_ratio or solidity < self.config.min_solidity:
            return None

        # LOCAL CONTRAST C: begin with an exact filled mask of the candidate interior.
        candidate_mask = np.zeros_like(gray)
        cv2.drawContours(candidate_mask, [contour], -1, 255, cv2.FILLED)

        # Dilation minus the original mask leaves only the ring immediately OUTSIDE
        # the candidate, preventing dark interior pixels from polluting I_outside.
        diameter = 2 * self.config.ring_width_px + 1
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        outer_ring = cv2.subtract(cv2.dilate(candidate_mask, ring_kernel), candidate_mask)
        inside = gray[candidate_mask > 0]
        outside = gray[outer_ring > 0]

        # Your formula: C = I_outside_ring - I_inside. Large positive = dark centre.
        contrast = float(outside.mean() - inside.mean()) if len(inside) and len(outside) else 0.0

        # BOUNDARY EDGE E: draw a narrow contour band and calculate what fraction of
        # that band overlaps strong Canny pixels.
        boundary = np.zeros_like(gray)
        cv2.drawContours(boundary, [contour], -1, 255, self.config.boundary_width_px)
        boundary_pixels = boundary > 0
        total_boundary_pixels = max(1, np.count_nonzero(boundary_pixels))
        strong_edge_pixels = np.count_nonzero((edges > 0) & boundary_pixels)
        edge_fraction = float(strong_edge_pixels / total_boundary_pixels)

        # C and E are both hard minimum filters and later contribute to ranking S.
        if contrast < self.config.min_local_contrast or edge_fraction < self.config.min_boundary_edge_fraction:
            return None

        # PHYSICAL FIT: the smaller bounding-box dimension is a conservative proxy
        # for opening diameter. For a final robot, prefer the largest inscribed circle
        # on a perspective-corrected depth/opening mask.
        diameter_px = min(width, height)
        diameter_cm = diameter_px / scale if scale else None
        required_cm = self.config.robot_diameter_cm * (1.0 + self.config.clearance_fraction)

        # None means "unknown", not "does not fit". Navigation should enter only on
        # explicit True plus separate depth, rim, slope, and approach-path checks.
        robot_fits = diameter_cm >= required_cm if diameter_cm is not None else None

        # AREA SCORE A: linear ramp reaching 1 when area is 4*min_area_px. Replace
        # this with a depth-aware preferred-size curve if field data justifies it.
        target_area = max(self.config.min_area_px * 4, 1.0)
        area_score = min(1.0, area / target_area)

        # Contour centroid is used for tracking and is better than box centre for an
        # irregular shape. Fall back to box centre only for a zero-area moment.
        moments = cv2.moments(contour)
        center = (
            (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            if moments["m00"]
            else (x + width / 2, y + height / 2)
        )

        # T and S begin at zero; tracker.update() and _score() fill them afterward.
        return Hole(
            contour=contour,
            bbox=(x, y, width, height),
            center=center,
            area_px=area,
            aspect_ratio=aspect,
            solidity=solidity,
            local_contrast=contrast,
            boundary_edge_fraction=edge_fraction,
            area_score=area_score,
            temporal_stability=0.0,
            score=0.0,
            diameter_cm=diameter_cm,
            robot_fits=robot_fits,
        )

    def _score(self, hole: Hole) -> float:
        """Implement S = wc*C + we*E + wa*A + wt*T using 0..1 terms."""
        weights = self.config.weights

        # Raw contrast uses grayscale units, so divide by 64 to map useful positive
        # contrast into 0..1. Change 64 if real measurements saturate too soon or
        # never approach 1. E, A, and T already have values in 0..1.
        contrast_score = min(1.0, max(0.0, hole.local_contrast / 64.0))
        return (
            weights.contrast * contrast_score
            + weights.edge * hole.boundary_edge_fraction
            + weights.area * hole.area_score
            + weights.temporal * hole.temporal_stability
        )
