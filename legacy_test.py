from ultralytics import YOLO

model = YOLO("best.pt")

results = model("dataset/test/test2", save=True)