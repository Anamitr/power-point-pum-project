from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import numpy as np

import image_util
import recognition_util


class CompleteClassifier:
    classifier: LinearSVC = None
    pca: PCA = None
    mean: np.ndarray = None
    std: np.ndarray = None

    def __init__(self, classifier: LinearSVC, pca: PCA, mean: np.ndarray, std: np.ndarray):
        self.classifier = classifier
        self.pca = pca
        self.mean = mean
        self.std = std

    def predict_one_image(self, image):
        features_normalized = recognition_util.normalize(image_util.getFeatures([image]), self.mean,
                                                         self.std)
        features_pca = self.pca.transform(features_normalized)
        return self.classifier.predict(features_pca)
