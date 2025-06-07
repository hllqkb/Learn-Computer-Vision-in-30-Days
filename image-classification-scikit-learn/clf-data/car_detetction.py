import pickle
import cv2
from skimage.transform import resize

model= pickle.load(open('./model.p','rb'))
image=cv2.imread('./clf-data/empty/00000000_00000161.jpg')
categories = ['empty', 'not_empty']

image1=resize(image,(15,15))
# if colorful image, convert to grayscale
if len(image1.shape)==3:
    image1=image1.mean(axis=2)
image1=image1.flatten().reshape(1,-1)
predictor=model.predict(image1)
# print(model.predict(cv2.imread('./test_image.jpg')))
print(categories[predictor[0]])
# # 预测概率（如果模型支持）
# if hasattr(model, "predict_proba"):
#     predicted_proba = model.predict_proba(image1)[0]
# else:
#     predicted_proba = None
# print(predicted_proba)