import joblib
import numpy as np
from models.feature_extractor import FeatureExtractor

class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, image_path):
        features = FeatureExtractor.extract_features(image_path).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        return "Real" if prediction == 0 else "Forged"
