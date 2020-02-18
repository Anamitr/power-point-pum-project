import os

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import image_util
import recognition_util
from CompleteClassifier import CompleteClassifier

print("Running Main2.py")

# once for each sign
# image_util.extract_folder_to_black_and_white(***)

# classic
images, labels = image_util.get_all_images_and_their_labels()
features = image_util.getFeatures(images)
# recognition_util.train_model_JN(features, labels)

# working
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
features_normalized = recognition_util.normalize(features, mean, std)
print("Normalized features")

pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_normalized)
print(pca.explained_variance_ratio_)

x_tr, x_tst, y_tr, y_tst = train_test_split(features_pca, labels, test_size=0.3)
model_SVC = SVC(kernel='linear', probability=True)
model_SVC.fit(x_tr, y_tr)
Z = model_SVC.predict(x_tst)
print(confusion_matrix(y_tst, Z))
print(accuracy_score(y_tst, Z, normalize=True))

complete_classifier = CompleteClassifier(model_SVC, pca, mean, std)

# before predicting - pca.transform

pass
