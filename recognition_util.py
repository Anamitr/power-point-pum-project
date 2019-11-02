from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from constants import *


def train_model(x, y):
    X_new = SelectKBest(f_classif, k=3).fit_transform(x, y)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y,
                                                        test_size=TRAINING_PART_RATIO, random_state=42)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    predicted_groups = classifier.predict(X_test)
    print(predicted_groups)
    print(y_test)
    pass

