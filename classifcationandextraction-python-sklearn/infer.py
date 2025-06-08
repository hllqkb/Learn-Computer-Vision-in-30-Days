from img2vec_pytorch import Img2Vec
import os
from PIL import Image
from joblib import load
import cv2
model=load('model.pkl')
img2vec=Img2Vec()
iamge_path=rf'D:\hllqkb\Pictures\90.jpg'
img=Image.open(iamge_path).convert('RGB')
features=img2vec.get_vec(img)
pred=model.predict([features])
print(pred)