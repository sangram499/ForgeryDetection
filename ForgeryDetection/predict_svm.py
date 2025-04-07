import cv2
import joblib

clf = joblib.load('models/svm_model.pkl')

def predict_single_image(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128, 128))
    features = img.flatten().reshape(1, -1)
    prediction = clf.predict(features)[0]
    return "Forged" if prediction == 1 else "Real"
