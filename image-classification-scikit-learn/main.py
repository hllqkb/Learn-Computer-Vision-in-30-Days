from skimage.io import imread
from skimage.transform import resize
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pickle
from sklearn.model_selection import GridSearchCV

# Prepare Data
input_dir = './clf-data/'
print("Input directory:", input_dir)
categories = ['empty', 'not_empty']
data = []
labels = []

for idx, category in enumerate(categories):
    path = os.path.join(input_dir, category)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory not found: {path}")
    
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = imread(img_path)
        img = resize(img, (15, 15))
        
        # Convert to grayscale if needed (remove channel dimension)
        if len(img.shape) == 3:
            img = img.mean(axis=2)  # or use img[:,:,0] for a specific channel
        
        data.append(img.flatten())  # Flatten to 1D array
        labels.append(idx)

data = np.array(data)
labels = np.array(labels)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
)

# Train Model
classifier = SVC()
parameters = [{'gamma': [0.001, 0.01, 0.1], 'C': [1, 10, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(X_train, y_train)

best_estimator = grid_search.best_estimator_
y_pred = best_estimator.predict(X_test)
score = accuracy_score(y_test, y_pred)
# print("Accuracy:", score)
print('{}% of samples were correctly classified.'.format(str(score*100)))
# Save Model
pickle.dump(best_estimator, open('./model.p', 'wb'))