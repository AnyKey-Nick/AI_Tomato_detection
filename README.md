# AI_Tomato_detection
AI-based tomato grading system.
This project uses a YOLOv11.

## Dataset
Dataset: "Tomates Computer Vision Project" from Roboflow  
Classes: `good_10_40`, `good_40_70`, `healthy`, `unripe`, `rotten`  
Images: 1025 annotated samples  
Split: train / valid / test

## Tools
- Python, OpenCV, Pandas
- YOLOv11 via Ultralytics
- Roboflow API
- EasyOCR, PyTesseract (optional)

## Model Training
Trained on 100 epochs  
Average mAP@0.5: **0.91**  
Best precision: over 95% for `unripe` and `good_40_70`  
Most confusion between visually similar classes

## Evaluation
PR, F1, Precision, Recall curves available in `/results`  
Confusion matrix visualized

## Presentation

The project presentation is available on [Canva]([https://www.canva.com/design/DAGopsSp9Fo/hXp6Wjo_Cd66MePSr-joJA/edit?utm_content=DAGopsSp9Fo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton](https://www.canva.com/design/DAGopsSp9Fo/hXp6Wjo_Cd66MePSr-joJA/view)


## Inference

**Image prediction:**
```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model("Tomato1.jpeg")
results[0].show()
```

**Video detection with object counting:**  
Run `src/detect_from_video.py`

## Install
```bash
pip install -r requirements.txt
```

## Author
Mykyta Anikieiev  


