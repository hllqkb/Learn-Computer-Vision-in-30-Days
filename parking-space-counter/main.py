from utils import empty_or_not, get_parking_spots_bboxes
import cv2
import numpy as np
import os
def calc_diff(img1,img2):
    return np.abs(img1.mean()-img2.mean())
local_path = os.path.dirname(os.path.abspath(__file__))
mask= cv2.imread(os.path.join(local_path, "mask_1920_1080.png"), 0)
video_path = os.path.join(local_path, "parking_1920_1080.mp4")
connect_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots=get_parking_spots_bboxes(connect_components)
cap=cv2.VideoCapture(video_path)
step=30
previous_frame=None
diffs=[None for i in spots]
spot_status=[None for j in spots]
frame_count = 0
while True:
    ret,frame=cap.read()
    
    if frame_count % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
    if frame_count % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot

            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

            isempty = empty_or_not(spot_crop)

            spot_status[spot_indx] = isempty
    if frame_count % step == 0:
        previous_frame = frame.copy()
    for idx,spot in enumerate(spots):
        x1, y1, w, h = spot
        # spot_bgr = frame[y1:y1 + h, x1:x1 + w]
        if spot_status[idx]:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 1)
    if ret:
        cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
        cv2.putText(frame,'Available Spots: {}/{}'.format(spot_status.count(True), len(spots)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    frame_count += 1