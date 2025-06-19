# Install required packages before running
# pip install roboflow ultralytics opencv-python pytesseract pandas easyocr

import os
import cv2
import numpy as np
import pytesseract
import pandas as pd
import random
import shutil
import easyocr
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO
from roboflow import Roboflow
from collections import defaultdict
from matplotlib import pyplot as plt

# Initialize Roboflow and download dataset
rf = Roboflow(api_key="your_key_here")  # Replace with your actual API key or use env variable
project = rf.workspace("tomatoes-xvztd").project("tomates-cel5z")
version = project.version(10)
dataset = version.download("yolov11")

# Define image folders
train_dir = os.path.join(dataset.location, "train", "images")
val_dir = os.path.join(dataset.location, "valid", "images")
test_dir = os.path.join(dataset.location, "test", "images")

# Train model
model = YOLO("yolov10n.pt")
results = model.train(data=dataset.location + "/data.yaml", epochs=100, imgsz=640)

# Load model for image inference
model = YOLO("C:/../best.pt")
image_path = "C:/.."  # Path to input image
image = cv2.imread(image_path)
assert image is not None, "Image not found!"
results = model(image)
annotated_frame = results[0].plot()
cv2.imwrite("C:/../name.jpg", annotated_frame)
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Load model for video inference
model = YOLO("C:/../weights/best.pt")
class_names = ['good_10_40', 'good_40_70', 'healthy', 'unripe', 'rotten']
cap = cv2.VideoCapture("video_tomato1.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output2.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

total_counts = defaultdict(int)
seen_objects = []
DIST_THRESHOLD = 50  # Pixel distance threshold

# Check if detected object is new
def is_new(center, label):
    for obj in seen_objects:
        if obj["label"] == label:
            prev_center = obj["center"]
            if np.linalg.norm(np.array(center) - np.array(prev_center)) < DIST_THRESHOLD:
                return False
    return True

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    frame_counts = defaultdict(int)

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        cls = int(cls)
        if cls >= len(class_names):
            continue
        label = class_names[cls]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center = (cx, cy)

        if is_new(center, label):
            total_counts[label] += 1
            seen_objects.append({"center": center, "label": label})
        frame_counts[label] += 1

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    y0 = 30
    for i, (label, count) in enumerate(frame_counts.items()):
        text = f"{label}: {count} (frame)"
        cv2.putText(frame, text, (10, y0 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Display total unique objects
    y1 = y0 + 25 * (len(frame_counts) + 1)
    cv2.putText(frame, "UNIQUE TOMATOES:", (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    for i, (label, total) in enumerate(total_counts.items()):
        text = f"{label}: {total}"
        cv2.putText(frame, text, (10, y1 + (i + 1)*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Detection", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Print total counts
print("\nTotal number of unique tomatoes by class:")
for label in class_names:
    print(f"{label}: {total_counts[label]}")
