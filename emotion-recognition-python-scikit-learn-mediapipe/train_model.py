import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import confusion_matrix
from sklearn.metrics import confusion_matrix

# Load data from train_data.txt
data = np.loadtxt('train_data.txt')

# Split features and labels (last column is the label)
X = data[:, :-1]  # Features (all columns except the last one)
y = data[:, -1]   # Labels (last column)

# Split data into training and testing sets (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y,shuffle=True)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set (optional)
y_pred = model.predict(X_test)

# Calculate accuracy (optional)
print("Unique labels in y:", np.unique(y))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")
print(confusion_matrix(y_test, y_pred))
# # Save the model (optional)
from joblib import dump
dump(model, 'random_forest_model.pkl')

