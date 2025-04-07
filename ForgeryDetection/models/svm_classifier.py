import os
import numpy as np
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.feature_extractor import FeatureExtractor

class SVMClassifier:
    def __init__(self, dataset_path, model_path):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.clf = svm.SVC(kernel='linear', probability=True)

    def load_data(self):
        X, y = [], []
        for label, category in enumerate(['real', 'forged']):
            folder = os.path.join(self.dataset_path, category)
            for file in os.listdir(folder):
                image_path = os.path.join(folder, file)
                features = FeatureExtractor.extract_features(image_path)
                X.append(features)
                y.append(label)
        
        return np.array(X), np.array(y)

    def train_model(self):
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, self.clf.predict(X_test))
        print(f"✅ Model trained with accuracy: {accuracy:.2f}")
        joblib.dump(self.clf, self.model_path)
        print(f"✅ Model saved at {self.model_path}")
