import os

import cv2
import numpy as np

import image_util
import recognition_util
from image_util import extract_folder_to_black_and_white

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


x, y = image_util.get_hu_moments()
recognition_util.train_model(x, y)

pass
