import cv2
import numpy as np

class FeatureExtractor:
    @staticmethod
    def extract_features(image_path):
        img = cv2.imread(image_path, 0)  # Read image in grayscale
        img = cv2.resize(img, (128, 128))  # Resize to fixed dimensions
        return img.flatten()  # Flatten into 1D array
