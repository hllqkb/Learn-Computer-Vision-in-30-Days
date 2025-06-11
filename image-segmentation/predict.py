
from ultralytics import YOLO
import os
import cv2
model_path = rf'C:\Users\hllqkb\Desktop\Learn-Computer-Vision-in-30-Days\Learn-Computer-Vision-in-30-Days\runs\segment\train\weights\best.pt'
image_path=rf'C:\Users\hllqkb\Desktop\Learn-Computer-Vision-in-30-Days\Learn-Computer-Vision-in-30-Days\Image-segmentation\KKs7NA69gl_small.jpg'
img=cv2.imread(image_path)
W,H,_=img.shape
model = YOLO(model_path)
results = model.predict(image_path)
for result in results:
    for j,mask in enumerate(result.masks.data):
        mask=mask.numpy()*255
        mask=cv2.resize(mask,(W,H))
        cv2.imwrite(os.path.join(os.path.dirname(image_path),f'mask_{j}.png'),mask)