from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from constants import *

import camera_controller
import image_util
import recognition_util
import tests

# prepare_images.rename_files()
# prepare_images.get_black_and_white_hand()
# image_util.check_cv_matching_shapes()
# prepare_images.test()

# camera_controller.start_camera()

x, y = image_util.get_hu_moments()
recognition_util.train_model(x, y)

# X_new = SelectKBest(chi2, k=3).fit_transform(x, y)
# X_train, X_test, y_train, y_test = train_test_split(X_new, y,
#                                                     test_size=TRAINING_PART_RATIO, random_state=42)
# classifier = KNeighborsClassifier(n_neighbors=3)
# classifier.fit(X_train, y_train)
# predicted_groups = classifier.predict(X_test)

pass


