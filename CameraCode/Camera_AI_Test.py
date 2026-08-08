from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.predict(source=1, show=True)