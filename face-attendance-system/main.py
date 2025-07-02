import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import starlette
import string
import random
import cv2
import pickle
import uuid
import datetime
import numpy as np
import shutil
import urllib
import face_recognition
ATTENDANCE_LOG_DIR = './logs'
DB_PATH = './db'
for dir_ in [ATTENDANCE_LOG_DIR, DB_PATH]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/login")
async def login(file: UploadFile=File(...)):
    file.name = f"{uuid.uuid4()}.jpg"
    context=await file.read()
    with open(os.path.join(file.name), 'wb') as f:
        f.write(context)
    img=cv2.imread(file.name)
    user_name,status=recognize(img)
    if status:
        epoch_time=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        date= datetime.datetime.now().strftime('%Y-%m-%d')
        log_file=os.path.join(ATTENDANCE_LOG_DIR, f"{date}.csv")
        with open(log_file, 'a') as f:
            f.write(f"{epoch_time},{user_name},IN\n")
            f.close()
    os.remove(file.name)
    return {'user': user_name,'status': status} 
@app.post("/register_new_user")
async def register_new_user(file: UploadFile=File(...), text=None):
    file.name = f"{uuid.uuid4()}.jpg"
    user_name=text
    context=await file.read()
    with open(os.path.join(file.name), 'wb') as f:
        f.write(context)
    shutil.copy(file.name, os.path.join(DB_PATH, f"{user_name}.jpg"))
    print(cv2.imread(file.name))
    embeeding = face_recognition.face_encodings(cv2.imread(file.name))
    with open(os.path.join(DB_PATH, f"{user_name}.pickle"), 'wb') as f:
        pickle.dump(embeeding, f)
    os.remove(file.name)
    return {'registration_status': 200}

@app.post("/logout")
async def logout(file: UploadFile=File(...)):
    file.name = f"{uuid.uuid4()}.jpg"
    context=await file.read()
    with open(os.path.join(file.name), 'wb') as f:
        f.write(context)
    img=cv2.imread(file.name)
    user_name,status=recognize(img)
    if status:
        epoch_time=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        date= datetime.datetime.now().strftime('%Y-%m-%d')
        log_file=os.path.join(ATTENDANCE_LOG_DIR, f"{date}.csv")
        with open(log_file, 'a') as f:
            f.write(f"{epoch_time},{user_name},OUT\n")
            f.close()
    return {'user': user_name,'status': status}


def recognize(img):
    embeedings = face_recognition.face_encodings(img)
    if len(embeedings) == 0:
        print("No face found in the image")
        return 'no_persons_found', False
    else:
        embeeding = embeedings[0]
    match = False
    i = 0
    db_dir = sorted([j for j in os.listdir(DB_PATH) if j.endswith('.pickle')])
    while (not match) and (i < len(db_dir)):
        path = os.path.join(DB_PATH, db_dir[i])
        with open(path, 'rb') as file:
            embeeded_list = pickle.load(file)
        # 跳过空或格式不对的 pickle 文件
        if not embeeded_list or len(embeeded_list[0]) != 128:
            i += 1
            continue
        embeeded = embeeded_list[0]
        match_result = face_recognition.compare_faces([np.array(embeeded)], np.array(embeeding), tolerance=0.6)
        match = match_result[0]
        i += 1
    if match:
        return db_dir[i-1].split('\\')[-1].split('.')[0], True
    else:
        return "Unknown", False