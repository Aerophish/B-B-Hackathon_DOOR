r"""One-shot visible camera scan returning candidate coordinate tuples.

Run from the project folder:

    .\venv\Scripts\python.exe run_camera.py

Output is always a tuple containing (x, y) tuples:

    ((x, y),)                      one confirmed candidate
    ((x1, y1), (x2, y2), ...)     multiple confirmed candidates
    ((-10000, -10000),)           no confirmed candidates

The image centre is (0, 0), positive x points right, and positive y points up.
"""

import argparse

import cv2

from hole_detection import DetectorConfig, HoleDetector


# CHANGE THIS VALUE to select the default camera:
#   0 = normally the built-in/first camera
#   1 = normally the second/USB camera
DEFAULT_CAMERA_INDEX = 0

# OPTIONAL CAMERA PREVIEW
# Set this to False if the Raspberry Pi runs without a display. You may also delete
# every block marked "OPTIONAL PREVIEW" below; detection and tuple output will keep
# working because preview code only draws a copy of each camera frame.
SHOW_CAMERA_PREVIEW = True

# A candidate needs 60 consecutive frames for confirmation. Scanning up to 120
# frames gives the camera time to initialise while keeping each run bounded.
DEFAULT_SCAN_FRAMES = 120

# Output is always a tuple OF (x, y) tuples, including the failure sentinel.
Coordinate = tuple[float, float]
ScanResult = tuple[Coordinate, ...]
NO_CANDIDATE: ScanResult = ((-10000.0, -10000.0),)


def _show_optional_preview(frame, holes) -> bool:
    """OPTIONAL PREVIEW: show the image and report whether Q/Esc was pressed.

    This affects display only. Set SHOW_CAMERA_PREVIEW=False, comment out the call
    in scan_camera(), or delete this function when running headless.
    """
    preview = frame.copy()

    # Only candidates that passed 60 consecutive frames receive a visible box.
    for hole in holes:
        if not hole.confirmed:
            continue
        x, y, width, height = hole.bbox
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 220, 0), 2)
        cv2.putText(
            preview,
            f"id={hole.track_id} score={hole.score:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        preview,
        "Scanning holes - Q/ESC to finish early",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("One-shot hole scan", preview)
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27)


def scan_camera(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    scan_frames: int = DEFAULT_SCAN_FRAMES,
) -> ScanResult:
    """Scan once and always return a tuple containing coordinate tuples.

    The full bounded scan allows candidates appearing slightly later to accumulate
    60 consecutive frames. Pressing Q/Esc ends it early and returns whatever is
    confirmed at that moment.
    """
    detector = HoleDetector(DetectorConfig())
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        return NO_CANDIDATE

    latest_confirmed = []
    final_frame_shape = None
    try:
        for _ in range(scan_frames):
            ok, frame = camera.read()
            if not ok:
                return NO_CANDIDATE

            holes = detector.detect(frame)
            latest_confirmed = [hole for hole in holes if hole.confirmed]
            final_frame_shape = frame.shape

            # ----- OPTIONAL PREVIEW START -----
            # Set SHOW_CAMERA_PREVIEW=False or remove this block for headless use.
            if SHOW_CAMERA_PREVIEW and _show_optional_preview(frame, holes):
                break
            # ----- OPTIONAL PREVIEW END -------
    finally:
        camera.release()

        # ----- OPTIONAL PREVIEW CLEANUP START -----
        if SHOW_CAMERA_PREVIEW:
            cv2.destroyAllWindows()
        # ----- OPTIONAL PREVIEW CLEANUP END -------

    if not latest_confirmed or final_frame_shape is None:
        return NO_CANDIDATE

    frame_height, frame_width = final_frame_shape[:2]

    # HoleDetector already sorts holes from highest to lowest score. Convert OpenCV
    # coordinates (top-left origin, y down) to camera-centred form (y up).
    coordinates: ScanResult = tuple(
        (
            float(hole.center[0] - frame_width / 2.0),
            float(frame_height / 2.0 - hole.center[1]),
        )
        for hole in latest_confirmed
    )

    # This remains a tuple OF tuples for one candidate: ((x, y),).
    return coordinates


def main() -> None:
    parser = argparse.ArgumentParser(description="Visibly scan once and return hole coordinate tuples.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"Camera number (default: {DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--scan-frames",
        type=int,
        default=DEFAULT_SCAN_FRAMES,
        help=f"Maximum frames to inspect (default: {DEFAULT_SCAN_FRAMES})",
    )
    args = parser.parse_args()

    # This is the only normal console output: always a tuple of (x, y) tuples.
    print(scan_camera(args.camera_index, args.scan_frames))


if __name__ == "__main__":
    main()
