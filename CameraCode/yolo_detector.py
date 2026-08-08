import cv2
import numpy as np
import onnxruntime as ort

from config import (
    MODEL_PATH,
    CONFIDENCE,
    YOLO_SIZE
)


# ============================================================
# LOAD YOLO ONNX
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("YOLO input:", input_name)
print("YOLO shape:", session.get_inputs()[0].shape)


# ============================================================
# YOLO INFERENCE
# ============================================================

def run_yolo(frame):
    """
    Run YOLO ONNX inference.

    Returns:
        detections

    Each detection is:
        [x1, y1, x2, y2, confidence, class_id]

    Coordinates are in the 640x640 model image.
    """

    img = cv2.resize(
        frame,
        (YOLO_SIZE, YOLO_SIZE)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(
        np.float32
    ) / 255.0

    # HWC -> CHW
    img = np.transpose(
        img,
        (2, 0, 1)
    )

    # Add batch dimension
    img = np.expand_dims(
        img,
        axis=0
    )

    outputs = session.run(
        None,
        {
            input_name: img
        }
    )

    return outputs[0][0]


# ============================================================
# GET BEST PERSON DETECTION
# ============================================================

def find_person(
    detections,
    image_width,
    image_height
):
    """
    Find the highest-confidence person detection.

    YOLO COCO class:
        0 = person

    Returns:
        target_pixel
        bounding_box
        confidence

    or:
        None, None, 0
    """

    best_conf = 0.0
    best_box = None

    for detection in detections:

        x1, y1, x2, y2, confidence, class_id = (
            detection
        )

        class_id = int(class_id)

        # Only interested in people
        if class_id != 0:
            continue

        if confidence < CONFIDENCE:
            continue

        if confidence <= best_conf:
            continue

        best_conf = float(
            confidence
        )

        # Convert 640x640 coordinates
        # back to camera coordinates

        x1 = x1 * image_width / YOLO_SIZE
        x2 = x2 * image_width / YOLO_SIZE

        y1 = y1 * image_height / YOLO_SIZE
        y2 = y2 * image_height / YOLO_SIZE

        best_box = (
            x1,
            y1,
            x2,
            y2
        )

    if best_box is None:

        return (
            None,
            None,
            0.0
        )

    x1, y1, x2, y2 = best_box

    target_pixel = np.array([
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0
    ])

    return (
        target_pixel,
        best_box,
        best_conf
    )