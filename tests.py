import random

import cv2

from image_util import get_typed_image_names, get_black_and_white_images


def check_cv_matching_shapes():
    # TODO: put data in dictionary
    black_and_white_images = get_black_and_white_images()

    for key in black_and_white_images.keys():
        distances = []
        for i in range(1, len(black_and_white_images[key])):
            distances.append(
                cv2.matchShapes(black_and_white_images[key][0], black_and_white_images[key][i], cv2.CONTOURS_MATCH_I2,
                                0),
            )
        print(key, ':', distances)

    print(
        cv2.matchShapes(black_and_white_images['left'][0], black_and_white_images['open'][0], cv2.CONTOURS_MATCH_I2, 0))
    print(
        cv2.matchShapes(black_and_white_images['play'][0], black_and_white_images['open'][0], cv2.CONTOURS_MATCH_I2, 0))

    return black_and_white_images

    for i in range(1, 5):
        print("Rand dist:" + str(cv2.matchShapes(black_and_white_images['close'][random.randint(0, 10)],
                                                 black_and_white_images['left'][random.randint(0, 10)],
                                                 cv2.CONTOURS_MATCH_I2,
                                                 0)))

    return


def test_hu_moments():
    black_and_white_images = get_black_and_white_images()
    num_of_same_type_to_print = 2
    for key in black_and_white_images.keys():
        print(key)
        for i in range(0, num_of_same_type_to_print):
            moments = cv2.moments(black_and_white_images[key][i])
            hu_moments = cv2.HuMoments(moments)
            print(hu_moments)
        print("\n\n")
