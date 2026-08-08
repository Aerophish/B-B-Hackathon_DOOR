import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Couldn't open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Couldn't read frame")
        break

    cv2.imshow("Endoscope", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

