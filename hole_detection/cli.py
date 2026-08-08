import argparse

import cv2

from .config import DetectorConfig
from .detector import HoleDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect dark hole candidates in an image.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pixels-per-cm", type=float)
    args = parser.parse_args()
    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Could not read {args.input}")
    detector = HoleDetector(DetectorConfig(pixels_per_cm=args.pixels_per_cm))
    holes = detector.detect(image)
    for hole in holes:
        x, y, w, h = hole.bbox
        colour = (0, 220, 0) if hole.robot_fits else (0, 165, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), colour, 2)
        label = f"{hole.score:.2f} id={hole.track_id} fit={hole.robot_fits}"
        cv2.putText(image, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1)
    cv2.imwrite(args.output, image)


if __name__ == "__main__":
    main()
