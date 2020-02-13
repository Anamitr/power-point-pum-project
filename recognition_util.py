import pickle

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

from constants import *


def train_model_with_knn(x, y):
    X_new = SelectKBest(f_classif, k=3).fit_transform(x, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y,
                                                        test_size=TRAINING_PART_RATIO, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    predicted_groups = classifier.predict(X_test)
    print(predicted_groups)
    print(y_test)
    print('Accuracy:', calculate_score(predicted_groups, y_test), '%')
    pass


def train_model_JN(x, y):
    features = x
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = normalize(features, mean, std)

    pca = PCA(n_components=0.95)
    features = pca.fit_transform(features)
    print(pca.explained_variance_ratio_)

    x_tr, x_tst, y_tr, y_tst = train_test_split(features, y, test_size=0.3)
    model_SVC = SVC(kernel='linear')
    model_SVC.fit(x_tr, y_tr)
    Z = model_SVC.predict(x_tst)
    print(confusion_matrix(y_tst, Z))
    print(accuracy_score(y_tst, Z, normalize=True))


def get_normalized_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return normalize(features, mean, std)


def normalize(data, mean, std):
    return (data - mean) / std


def calculate_score(predicted, actual):
    if len(predicted) != len(actual):
        print("calculate_score: length of arguments must be even!")
        return
    else:
        same = 0
        for i in range(len(predicted)):
            if predicted[i] == actual[i]:
                same += 1
        return same / len(predicted) * 100


def save_model(save_name: str, model):
    with open(save_name, 'wb') as fid:
        pickle.dump(model, fid)


def open_model(save_name: str):
    with open(save_name, 'rb') as fid:
        return pickle.load(fid)
