import os

import cv2
import numpy as np

from constants import BASE_PATH, BASE_IMAGE_EXTENSION, TYPES_OF_GESTURES


def rename_files():
    path = "db"

    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    print(subfolders)

    for folder in subfolders:
        files = os.listdir(folder)
        i = 1
        for file in files:
            src = folder + '\\' + file
            dst = folder + '\\' + str(i) + '.jpg'
            os.rename(src, dst)
            i = i + 1


def get_black_and_white_hand_from_path(img_path):
    image = cv2.imread(img_path)
    return get_black_and_white_hand(image)


def get_black_and_white_hand(image):
    h1_mask = get_image_mask(image)

    contours, hierarchy = cv2.findContours(h1_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # [print(len(contour)) for contour in contours]
    longest_contour = get_longest_contour(contours)
    # print(len(longest_contour))
    new_contours = [longest_contour]

    h1_contours = np.copy(image)

    black_and_white_img = np.zeros_like(h1_contours)

    try:
        # cv2.drawContours(h1_contours, new_contours, -1, (0, 0, 255), 20, hierarchy=hierarchy, maxLevel=0)
        cv2.drawContours(black_and_white_img, new_contours, 0, (255, 255, 255), -1)
    except cv2.error:
        print("Not able to draw contours!")

    # to have single channel image, as required by function matchShapes
    black_and_white_img = cv2.cvtColor(black_and_white_img, cv2.COLOR_BGR2GRAY)
    # show_image(black_and_white_img)

    return black_and_white_img


def get_longest_contour(contours):
    max_length = -1
    longest_contour = None
    for contour in contours:
        if len(contour) > max_length:
            longest_contour = contour
            max_length = len(contour)
    return longest_contour


def get_image_mask(h1):
    # zmiana przestrzeni kolorów z BGR do HSV
    h1_hsv = cv2.cvtColor(h1, cv2.COLOR_BGR2HSV)

    # określenie zakresu kolorów, które nas interesują (w przestrzeni HSV)
    lower = np.array([40, 40, 40])
    upper = np.array([70, 255, 255])

    # Progowanie obrazu za pomocą zdefiniowanych zakresów
    h1_mask = cv2.inRange(h1_hsv, lower, upper)
    # show_image(h1_mask)
    return h1_mask


def get_black_and_white_images():
    img_typed_resources = get_typed_image_names()
    black_and_white_images = dict()
    for key in img_typed_resources.keys():
        images = []
        for img_num in img_typed_resources[key]:
            black_and_white_hand = get_black_and_white_hand_from_path(get_img_path_from_img_type_and_num(key, img_num))

            images.append(black_and_white_hand)
            # if counter == 4:
            #     break
        black_and_white_images[key] = images
    return black_and_white_images


def get_typed_image_names():
    img_typed_resources = dict()
    for gesture_type in TYPES_OF_GESTURES:
        images = []
        for r, d, f in os.walk(BASE_PATH + gesture_type):
            for file in f:
                images.append(int(file.replace(BASE_IMAGE_EXTENSION, '')))

        images.sort()
        img_typed_resources[gesture_type] = images
        # print(gesture_type + ': ', images)
    # print(img_typed_resources)
    return img_typed_resources


def get_hu_moments():
    black_and_white_images = get_black_and_white_images()
    # num_of_images = len(black_and_white_images)
    num_of_hu_moments = 7

    # hu_moments_list = np.zeros([count_len_of_images_dict(images_dict=black_and_white_images), num_of_hu_moments], float)
    hu_moments_list = np.empty((0, num_of_hu_moments), float)
    # hu_moments_list = []
    labels = []
    for key in black_and_white_images.keys():
        # print(key)
        for image in black_and_white_images[key]:
            hu_moments = cv2.HuMoments(cv2.moments(image))
            # print(hu_moments)
            hu_moments_list = np.append(hu_moments_list, hu_moments.T, axis=0)
            # hu_moments_list.append(hu_moments.tolist())
            labels.append(key)

    return hu_moments_list, np.array(labels).T
    # return np.array(hu_moments_list), np.array(labels).T


def count_len_of_images_dict(images_dict):
    c = 0
    for key in images_dict.keys():
        for item in images_dict[key]:
            c += 1
    return c


def show_image(image):
    try:
        cv2.imshow("hand1", cv2.resize(image, (300, 400)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("Not able to show image!")


def get_img_path_from_img_type_and_num(type, num):
    return BASE_PATH + str(type) + '/' + str(num) + BASE_IMAGE_EXTENSION
