from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
import numpy as np

import image_util
import recognition_util


class CompleteClassifier:
    classifier: SVC = None
    pca: PCA = None
    mean: np.ndarray = None
    std: np.ndarray = None

    def __init__(self, classifier: LinearSVC, pca: PCA, mean: np.ndarray, std: np.ndarray):
        self.classifier = classifier
        self.pca = pca
        self.mean = mean
        self.std = std

    def predict_one_image(self, image):
        features_pca = self.get_features_pca(image)
        return self.classifier.predict(features_pca)

    def get_predict_proba_of_image(self, image):
        features_pca = self.get_features_pca(image)
        return self.classifier.predict_proba(features_pca)

    def get_features_pca(self, image):
        return self.pca.transform(recognition_util.normalize(image_util.get_features([image]), self.mean, self.std))
