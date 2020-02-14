import os

import cv2
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

import constants
import image_util
import recognition_util
from CompleteClassifier import CompleteClassifier
from constants import PROJECT_NAME
import constants

complete_classifier: CompleteClassifier = recognition_util.open_model('trained_models/complete_classifier_2')


def start_camera():
    camera = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')

    while True:
        (grabbed, frame) = camera.read()
        cv2.imshow(PROJECT_NAME, frame)

        black_and_white_image = image_util.get_black_and_white_hand(frame)

        try:
            if check_if_applies_to_threshold(black_and_white_image):
                print("Applies to threshold", complete_classifier.predict_one_image(black_and_white_image), ', ',
                      complete_classifier.get_predict_proba_of_image(black_and_white_image)[0])
        except ValueError as valueError:
            print('ValueError', valueError)

        # print("predicted:", complete_classifier.predict_one_image(black_and_white_image))
        # print("proba:", complete_classifier.get_predict_proba_of_image(black_and_white_image)[0])
        # print("applies:", check_if_applies_to_threshold(black_and_white_image))

        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(os.path.join('temp', "temp.jpg"), black_and_white_image)
            print("predicted:", complete_classifier.predict_one_image(black_and_white_image))
            print("proba:", complete_classifier.get_predict_proba_of_image(black_and_white_image)[0])
            print("applies:", check_if_applies_to_threshold(black_and_white_image))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def check_if_applies_to_threshold(black_and_white_image):
    return True in [item > constants.CLASSIFICATION_PROBABILITY_THRESHOLD for item in
                    complete_classifier.get_predict_proba_of_image(black_and_white_image)[0]]


# Main script
start_camera()
