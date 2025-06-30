import pickle
import cv2
import os
import mediapipe as mp
import matplotlib.pyplot as plt
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands= mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)
local_path = os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(local_path,'data')
data=[]
label=[]
for dirs_ in os.listdir(DATA_DIR):
    if dirs_.startswith('.'):
        continue
    dir_path = os.path.join(DATA_DIR, dirs_)
    for file_name in os.listdir(dir_path):
        data_aux=[]
        x_=[]
        y_=[]
        if file_name.startswith('.'):
            continue
        file_path = os.path.join(dir_path, file_name)
        image = cv2.imread(file_path)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            continue
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)
            for i in range(len(hand_landmarks.landmark)):
                x=hand_landmarks.landmark[i].x
                y=hand_landmarks.landmark[i].y
                data_aux.append(x-min(x_))
                data_aux.append(y-min(y_))
            data.append(data_aux)
            label.append(dirs_)
            # drawing the landmarks on the image (optional)
        #     mp_drawing.draw_landmarks(
        #         image=image,
        #         landmark_list=hand_landmarks,
        #         connections=mp_hands.HAND_CONNECTIONS,
        #         landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
        #         connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        # plt.imshow(image)
        # plt.show()
print("everything is ok")
f=open(os.path.join(local_path,'data.pickle'),'wb')
pickle.dump({'data':data,'label':label},f)
f.close()