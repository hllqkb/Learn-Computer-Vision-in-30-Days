import pickle
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
local_path = os.path.dirname(os.path.abspath(__file__))

# Load the data
data_pickle=pickle.load(open(os.path.join(local_path, 'data.pickle'), 'rb'))
data=np.array(data_pickle['data'])
labels=np.array(data_pickle['label'])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=0.2, random_state=42,shuffle=True)
model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Evaluate the model on the testing set
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f=open(os.path.join(local_path, 'model.p'), 'wb')
pickle.dump({'model':model}, f)
f.close()