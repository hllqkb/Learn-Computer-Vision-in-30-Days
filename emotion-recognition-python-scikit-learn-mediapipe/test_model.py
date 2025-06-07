import joblib
from utils import get_face_landmarks
import numpy as np
import cv2
import sys
import os
emotions=['angry','sad','surprised']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告日志
# Load the model
model = joblib.load('./random_forest_model.pkl')
# print(list(model.signatures.keys()))  # 查看所有签名
# sys.exit()
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
while ret:
    ret,frame=cap.read()
    if ret:
        
        landmarks = get_face_landmarks(frame,static_image_mode=False,draw=True)
        output = model.predict([landmarks])
        cv2.putText(frame, emotions[int(output[0])], (10, frame.shape[0]-1), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

        cv2.imshow('frame',frame)
        # print(output)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
cv2.destroyAllWindows()
cap.release()

