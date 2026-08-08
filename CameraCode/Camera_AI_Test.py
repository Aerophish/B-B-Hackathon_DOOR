import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("yolo26n.onnx")
input_name = session.get_inputs()[0].name

# Camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera")
        break

    # Original image dimensions
    h, w = frame.shape[:2]

    # Prepare image for YOLO
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Run inference
    outputs = session.run(None, {input_name: img})
    detections = outputs[0][0]

    # Draw detections
    for detection in detections:

        x1, y1, x2, y2, confidence, class_id = detection

        if confidence < 0.5:
            continue

        # Convert coordinates from 640x640 → camera resolution
        x1 = int(x1 * w / 640)
        x2 = int(x2 * w / 640)
        y1 = int(y1 * h / 640)
        y2 = int(y2 * h / 640)

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"Class {int(class_id)}: {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Show image
    cv2.imshow("YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()