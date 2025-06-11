import os
from ultralytics import YOLO
model=YOLO('./yolov8n-seg.pt')
results = model.train(data="coco8-seg.yaml", epochs=10, imgsz=640)