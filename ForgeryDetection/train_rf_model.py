import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
os.makedirs('models', exist_ok=True)

folder = 'data/raw_edited_pairs'
image_pairs = [('real1.jpg', 'fake1.jpg')]

X, y = [], []

for real_img, fake_img in image_pairs:
    img1 = cv2.imread(os.path.join(folder, real_img))
    img2 = cv2.imread(os.path.join(folder, fake_img))
    if img1 is None or img2 is None: continue

    diff = cv2.absdiff(img1, img2)
    diff = cv2.resize(diff, (64, 64))
    X.append(diff.flatten())
    y.append(1)

    same = cv2.absdiff(img1, img1)
    same = cv2.resize(same, (64, 64))
    X.append(same.flatten())
    y.append(0)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

joblib.dump(clf, 'models/rf_model.pkl')
print("✅ Random Forest model saved!")
