from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import cv2

# Load your trained model
model = YOLO("yolov8n.pt")   # Load an official YOLOv8 model

def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        return

    img = cv2.imread(file_path)

    results = model(img)

    annotated = results[0].plot()  # draw boxes

    cv2.imshow("Pothole Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


root = tk.Tk()
root.title("Pothole Detection Test")

upload_btn = tk.Button(root, text="Upload Image", command=upload_and_predict)
upload_btn.pack(pady=20)

root.mainloop()