import os

import cv2
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

import constants
import image_util
import recognition_util
from CompleteClassifier import CompleteClassifier
from constants import PROJECT_NAME

complete_classifier: CompleteClassifier = recognition_util.open_model('trained_models/complete_classifier_1')


def start_camera():
    camera = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')

    while True:
        (grabbed, frame) = camera.read()
        cv2.imshow(PROJECT_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            black_and_white_image = image_util.get_black_and_white_hand(frame)
            cv2.imwrite(os.path.join('temp', "temp.jpg"), black_and_white_image)

            print("predicted:", complete_classifier.predict_one_image(black_and_white_image))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Main script
start_camera()
