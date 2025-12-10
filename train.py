from ultralytics import YOLO

# --------------------------------
DATA_YAML = "pothole.yaml"     # your dataset YAML
PRETRAINED = "yolov8s.pt"      # model to start with
# --------------------------------

def train():
    model = YOLO(PRETRAINED)

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=4,
        workers=0,
        device="cpu"         # 👈 FORCE CPU
    )

    print("\nTraining Completed on CPU!")
    print("Model saved at: runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    train()
