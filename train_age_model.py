
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# --- Dataset 1 ---
csv1 = os.path.join('Dataset', 'Dataset 1', 'faces', 'train.csv')
img_dir1 = os.path.join('Dataset', 'Dataset 1', 'faces', 'Train')
df1 = pd.read_csv(csv1)

# --- Dataset 2 ---
csv2 = os.path.join('Dataset', 'Dataset 2', 'age_detection.csv')
df2 = pd.read_csv(csv2)

X = []
y = []

print(f"Loaded Dataset 1: {len(df1)} samples, Dataset 2: {len(df2)} samples")


# Map YOUNG/MIDDLE/OLD to year ranges
def map_class_to_age(label):
    # Map YOUNG/MIDDLE/OLD to new bins
    if label == 'YOUNG':
        return '18-25'  # best guess for young
    elif label == 'MIDDLE':
        return '35-45'  # best guess for middle
    elif label == 'OLD':
        return '55-65'  # best guess for old
    else:
        return label

# Load Dataset 1 (labels: 'Class' - YOUNG/MIDDLE/OLD)
for idx, row in df1.iterrows():
    img_path = os.path.join(img_dir1, str(row['ID']))
    label = map_class_to_age(str(row['Class']))
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (32, 32))
            X.append(img.flatten())
            y.append(label)


# Map age ranges to unified bins
def map_age_to_bin(age):
    # Map age ranges to new bins
    age = str(age)
    if age in ['18-20', '21-30', '18-25']:
        return '18-25'
    elif age in ['25-35']:
        return '25-35'
    elif age in ['31-40', '35-45']:
        return '35-45'
    elif age in ['41-50', '45-55']:
        return '45-55'
    elif age in ['51-60', '55-65']:
        return '55-65'
    else:
        # Try to parse numeric age
        try:
            age_num = int(age)
            if 18 <= age_num <= 25:
                return '18-25'
            elif 25 < age_num <= 35:
                return '25-35'
            elif 35 < age_num <= 45:
                return '35-45'
            elif 45 < age_num <= 55:
                return '45-55'
            elif 55 < age_num <= 65:
                return '55-65'
        except:
            pass
        return age

# Load Dataset 2 (labels: 'age' - 18-20, 21-30, ...)
for idx, row in df2.iterrows():
    img_path = os.path.join('Dataset', 'Dataset 2', row['file'])
    label = map_age_to_bin(str(row['age']))
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            # Histogram equalization for better contrast
            img = cv2.equalizeHist(img)
            X.append(img.flatten())
            y.append(label)

print(f"Total images loaded: {len(X)}")
print(f"Unique labels: {set(y)}")

X = np.array(X)
y = np.array(y)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train improved model
clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))

# Show confusion matrix for more insight
from sklearn.metrics import confusion_matrix, classification_report
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'age_model.pkl')
print('Model saved as age_model.pkl')
