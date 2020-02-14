import os

import cv2
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from pynput.keyboard import Key, Controller

import constants
import image_util
import recognition_util
from CompleteClassifier import CompleteClassifier
from constants import PROJECT_NAME
import constants

complete_classifier: CompleteClassifier = recognition_util.open_model('trained_models/complete_classifier_2')
keyboard = Controller()

def start_camera():
    camera = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')
    occured_sign_list = []

    while True:
        (grabbed, frame) = camera.read()
        cv2.imshow(PROJECT_NAME, frame)

        black_and_white_image = image_util.get_black_and_white_hand(frame)

        try:
            if check_if_applies_to_threshold(black_and_white_image):
                predicted_sign = complete_classifier.predict_one_image(black_and_white_image)[0]
                print("Applies to threshold", predicted_sign, ', ',
                      complete_classifier.get_predict_proba_of_image(black_and_white_image)[0])
                occured_sign_list.append(predicted_sign)
                if len(occured_sign_list) >= constants.SIGN_REPETITION_THRESHOLD:
                    if len(set(occured_sign_list)) == 1:
                        perform_action(predicted_sign)
                        occured_sign_list.clear()
                    else:
                        occured_sign_list.pop(0)
        except ValueError as valueError:
            print('ValueError', valueError)
        except IndexError as indexError:
            print('IndexError', indexError)

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


def perform_action(action: str):
    print('Performing action', action)
    if action == 'left':
        keyboard.press(Key.right)
        keyboard.release(Key.right)
    elif action == 'right':
        keyboard.press(Key.left)
        keyboard.release(Key.left)
    else:
        print('Unrecognized action:', action)


# Main script
start_camera()
