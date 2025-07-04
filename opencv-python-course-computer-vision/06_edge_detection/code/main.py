import os

import cv2
import numpy as np


img = cv2.imread(os.path.join('./hllqk.jpg'))
# Cv2.Canny Edges detection
img_edge = cv2.Canny(img, 200, 300)
# 膨胀边缘dilate
img_edge_d = cv2.dilate(img_edge, np.ones((4, 4), dtype=np.int8))
# 腐蚀边缘erode
img_edge_e = cv2.erode(img_edge_d, np.ones((10, 20), dtype=np.int8))

cv2.imshow('img', img)
cv2.imshow('img_edge', img_edge)
cv2.imshow('img_edge_d', img_edge_d)
cv2.imshow('img_edge_e', img_edge_e)
cv2.waitKey(0)
