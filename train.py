from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt") # change to best
    model.train(
    data="dataset2/data.yaml",
    epochs=100,
    device=None,
    workers=8,
    batch=-1,
    imgsz=1280,

    # augmentations
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.3,
    scale=0.5,
    translate=0.1,
    fliplr=0.5,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,


    box=10.0,
)
if __name__ == "__main__":
    main()
