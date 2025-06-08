from img2vec_pytorch import Img2Vec
import os
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
## Using Img2Vec to extract image features and train a Random Forest classifier
# Prepare Data
img2vec = Img2Vec()
data_dir = r'C:\Users\hllqkb\Desktop\Learn-Computer-Vision-in-30-Days\Learn-Computer-Vision-in-30-Days\classifcationandextraction-python-sklearn\dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}

for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        category_dir = os.path.join(dir_, category)
        if not os.path.isdir(category_dir):  # 跳过非目录文件
            continue
        for img_path in os.listdir(category_dir):
            img_path_ = os.path.join(category_dir, img_path)
            try:
                img = Image.open(img_path_).convert('RGB')
                feature = img2vec.get_vec(img)
                features.append(feature)
                labels.append(category)
            except Exception as e:
                print(f"Skipped {img_path_} due to error: {e}")
    data[['train_data', 'val_data'][j]] = features
    data[['train_labels', 'val_labels'][j]] = labels

# print(data.keys())
# Train Model
model=RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(data['train_data'], data['train_labels'])

# Test Performance
y_pred = model.predict(data['val_data'])
score=accuracy_score( y_pred,  data['val_labels'])
print(f"Accuracy: {score*100:.2f}%")
# Save Model
import joblib
joblib.dump(model, 'model.pkl')