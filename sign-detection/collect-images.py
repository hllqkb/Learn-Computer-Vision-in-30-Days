import cv2
import os
number_of_classes= 3
data_size= 100
local_path= os.path.dirname(os.path.abspath(__file__))
data_dir= os.path.join(local_path, 'data')
print(os.makedirs(name=data_dir, mode=0o777, exist_ok=True))
cap = cv2.VideoCapture(0)
for i in range(number_of_classes):
    if not os.path.exists(os.path.join(data_dir, str(i))):
        os.makedirs(os.path.join(data_dir, str(i)))
    print('Collecting data for class {}'.format(i))
    done = False
    while True:
        ret,frame=cap.read()
        cv2.putText(frame, 'Press "Q" to Start :)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    counter=0
    while counter < data_size:
        ret, frame = cap.read()
        cv2.waitKey(50)
        cv2.putText(frame, 'Collecting data for class {}: {}/{}'.format(i, counter + 1, data_size), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(data_dir, str(i), '{}.jpg'.format(counter)), frame)
        counter += 1
    print('Data collection for class {} completed'.format(i))
        
cap.release()
cv2.destroyAllWindows()