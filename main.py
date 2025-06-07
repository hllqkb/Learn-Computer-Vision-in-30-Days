import cv2
import os
img_path=os.path.join(os.getcwd(),'hllqk.jpg')
img=cv2.imread(img_path)
# cv2.imwrite(os.path.join(os.getcwd(),'test_copy.jpg'),img)
k_size=7
img_blur=cv2.blur(img,(k_size,k_size))
cv2.imshow('blur',img_blur)
# img=img[100:300,100:300]
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()