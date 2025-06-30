from utils import empty_or_not, get_parking_spots_bboxes
import cv2
import numpy as np
import os
local_path = os.path.dirname(os.path.abspath(__file__))
mask= cv2.imread(os.path.join(local_path, "mask_crop.png"), 0)
video_path = os.path.join(local_path, "parking_crop_loop.mp4")
cap=cv2.VideoCapture(video_path)
while True:
    ret,frame=cap.read()
    if ret:
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break