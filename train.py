from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="dataset2/data.yaml",
        epochs=50,
        imgsz=640,
        device=0, #  set to 'mps' on apple silicon, and 'cpu' otherwise. 0 means GPU
        workers=18,
        batch=24
    )

if __name__ == "__main__":
    main()
