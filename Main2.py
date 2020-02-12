import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import image_util
import recognition_util

print("Running Main2.py")

# image = cv2.imread('db2/close/2.jpg')
# cv2.imshow('test image', image)
# image_util.show_image(image)

# tests.test_hu_moments()
db_path = "db3"
sign_path = "open"

path = db_path + "/" + sign_path

print(os.listdir(path))

# extract_folder_to_black_and_white(path)

# TYPES_OF_GESTURES = ['open', 'close', 'right', 'left', 'play']
#
# black_and_white_images = dict()
#
# for file_name in os.listdir(path):
#     full_path = os.path.join(path, file_name)
#     black_and_white_images.append(cv2.imread(full_path))
#
# num_of_hu_moments = 7
#
# # hu_moments_list = np.zeros([count_len_of_images_dict(images_dict=black_and_white_images), num_of_hu_moments], float)
# hu_moments_list = np.empty((0, num_of_hu_moments), float)
# # hu_moments_list = []
# labels = []
# for key in black_and_white_images.keys():
#     # print(key)
#     for image in black_and_white_images[key]:
#         hu_moments = cv2.HuMoments(cv2.moments(image))
#         # print(hu_moments)
#         hu_moments_list = np.append(hu_moments_list, hu_moments.T, axis=0)
#         # hu_moments_list.append(hu_moments.tolist())
#         labels.append(key)
#
# hu_moments_list, np.array(labels).T


# x, y = image_util.get_hu_moments()
# x, y = image_util.get_featues_and_labels()
# recognition_util.train_model_with_knn(x, y)
# recognition_util.train_model_JN(x, y)

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
model_SVC = SVC(kernel='linear')
model_SVC.fit(x_tr, y_tr)
Z = model_SVC.predict(x_tst)
print(confusion_matrix(y_tst, Z))
print(accuracy_score(y_tst, Z, normalize=True))

pass
