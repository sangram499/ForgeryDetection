import cv2
import numpy as np
import joblib

model = joblib.load('models/rf_model.pkl')

def predict_forgery(real_path, edited_path):
    real = cv2.imread(real_path)
    edited = cv2.imread(edited_path)
    if real is None or edited is None:
        return "Invalid image(s)"
    diff = cv2.absdiff(real, edited)
    diff = cv2.resize(diff, (64, 64))
    features = diff.flatten().reshape(1, -1)
    pred = model.predict(features)[0]
    return "Forged" if pred == 1 else "Not Forged"
