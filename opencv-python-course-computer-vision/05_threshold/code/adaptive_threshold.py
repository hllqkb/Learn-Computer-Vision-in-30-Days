import os

import cv2
# 阀值化可以将灰度图像转为二值图像yes or no，阀值化的目的是将图像中明显的边缘和噪声分离开来，从而使图像中有用的信息更加突出。
# 阀值化的原理是将图像中的像素值与一个阈值进行比较，大于阈值的像素值为白色，小于阈值的像素值为黑色。
# 阀值化的目的是将图像中明显的边缘和噪声分离开来，从而使图像中有用的信息更加突出。

img = cv2.imread(os.path.join(os.getcwd(),'opencv-python-course-computer-vision/05_threshold/code', 'handwritten.png'))
# //opencv-python-course-computer-vision/05_threshold/code/adaptive_threshold.py

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 自适应filters的阈值化
adaptive_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)
# 简单阈值化
ret, simple_thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)


cv2.imshow('img', img)
cv2.imshow('adaptive_thresh', adaptive_thresh)
cv2.imshow('simple_thresh', simple_thresh)
cv2.waitKey(0)