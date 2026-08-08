"""Live camera runner for the modular hole detector.

Run with:
    python -m hole_detection.camera

Press Q or Escape in the preview window to stop safely.
"""

import argparse
import time
from typing import Optional

import cv2

from .config import DetectorConfig
from .detector import HoleDetector
from .models import Hole


def _fit_label(robot_fits: Optional[bool]) -> str:
    """Keep unknown physical scale visually distinct from a definite rejection."""
    if robot_fits is True:
        return "FITS"
    if robot_fits is False:
        return "TOO SMALL"
    return "FIT UNKNOWN"


def _colour(hole: Hole) -> tuple[int, int, int]:
    """Green=fit, orange=too small, cyan=scale unknown (all colours are BGR)."""
    if hole.robot_fits is True:
        return (0, 220, 0)
    if hole.robot_fits is False:
        return (0, 140, 255)
    return (255, 220, 0)


def annotate(frame, holes: list[Hole], fps: float):
    """Draw only candidates confirmed for the configured persistence period."""
    output = frame.copy()
    confirmed_holes = [hole for hole in holes if hole.confirmed]

    # Deliberately do not draw tentative candidates: rapidly appearing/disappearing
    # boxes are distracting and unsafe for a future navigation consumer.
    for hole in confirmed_holes:
        x, y, width, height = hole.bbox
        colour = _colour(hole)
        cv2.rectangle(output, (x, y), (x + width, y + height), colour, 2)

        diameter = f"{hole.diameter_cm:.1f}cm" if hole.diameter_cm is not None else "uncalibrated"
        label = (
            f"id={hole.track_id} S={hole.score:.2f} "
            f"{_fit_label(hole.robot_fits)} {diameter}"
        )
        cv2.putText(
            output,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            colour,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        f"FPS {fps:.1f} | confirmed {len(confirmed_holes)} | validating {len(holes) - len(confirmed_holes)} | Q/ESC to exit",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hole detection on a live camera.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index; usually 0")
    parser.add_argument(
        "--pixels-per-cm",
        type=float,
        default=None,
        help="Calibrated scale at the observed surface; omit to show FIT UNKNOWN",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    detector = HoleDetector(DetectorConfig(pixels_per_cm=args.pixels_per_cm))
    camera = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        camera.release()
        raise SystemExit(
            f"Could not open camera index {args.camera_index}. "
            "Close other camera apps or try --camera-index 1."
        )

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    print("Camera opened. Focus the preview window and press Q or Escape to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera opened but stopped returning frames.")

            holes = detector.detect(frame)
            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            smoothed_fps = instantaneous_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instantaneous_fps

            cv2.imshow("Hole detection - live camera", annotate(frame, holes, smoothed_fps))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
