import cv2
import numpy as np
import os
from utils import get_face_landmarks
train_dara='./train'
outputfile='train_data.txt'
max_samples=1000
with open(outputfile,'w') as f:
    for index,emotion in enumerate(sorted(os.listdir(train_dara))):
        sample_count= 0
        if not os.path.isdir(os.path.join(train_dara,emotion)):
            continue
        for img_path_ in os.listdir(os.path.join(train_dara, emotion)):
            sample_count+=1
            if sample_count>max_samples:
                break
            img_path=os.path.join(train_dara, emotion, img_path_)
            image=cv2.imread(img_path)
            face_landmarks=get_face_landmarks(image)
            if len(face_landmarks)==1404:
                face_landmarks.append(int(index))
 
                np.savetxt(f,[np.asarray(face_landmarks)])
        # print(index)
    
                # output.append(face_landmarks)