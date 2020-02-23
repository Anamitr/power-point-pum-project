import os

import cv2
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from pynput.keyboard import Key, Controller
import pynput
from subprocess import call

import constants
import image_util
import recognition_util
from CompleteClassifier import CompleteClassifier
from constants import PROJECT_NAME
import constants

SHOULD_ACTUALLY_PERFORM_ACTIONS = True

complete_classifier: CompleteClassifier = recognition_util.open_model('trained_models/complete_classifier_8')
keyboard = Controller()
mouse = pynput.mouse.Controller()
volume = 50
pointer_on = False


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
                # print("Applies to threshold", predicted_sign, ', ',
                #       complete_classifier.get_predict_proba_of_image(black_and_white_image)[0])
                occured_sign_list.append(predicted_sign)
                if len(occured_sign_list) >= constants.POINTER_SIGN_REPETITION_THRESHOLD:
                    if len(set(occured_sign_list)) == 1 and predicted_sign == 'pointer2':
                        handle_pointer(black_and_white_image)
                if len(occured_sign_list) >= constants.SIGN_REPETITION_THRESHOLD:
                    if len(set(occured_sign_list)) == 1:
                        print('Applying move', predicted_sign)
                        if SHOULD_ACTUALLY_PERFORM_ACTIONS:
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
    global pointer_on
    if action != 'pointer2' and pointer_on is True:
        keyboard.press('l')
        keyboard.release('l')
        pointer_on = False

    print('Performing action', action)
    if action == 'left':
        keyboard.press(Key.right)
        keyboard.release(Key.right)
    elif action == 'right':
        keyboard.press(Key.left)
        keyboard.release(Key.left)
    elif action == 'play2':
        keyboard.press('k')
        keyboard.release('k')
    elif action == 'volume_up':
        raise_volume()
        # keyboard.press(Key.media_volume_up)
        pass
    elif action == 'volume_down2':
        lower_volume()
        # keyboard.press(Key.media_volume_down)
    elif action == 'pointer2':
        pass
    else:
        print('Unrecognized action:', action)


def handle_pointer(baw_image):
    global pointer_on
    if pointer_on is False:
        keyboard.press('l')
        keyboard.release('l')
        pointer_on = True
    relative_point = image_util.get_farther_left_point_on_baw_img(baw_image)
    x = relative_point[0]
    x = (x - 0.5) * 2
    y = relative_point[1]
    y = (y - 0.5) * 2
    mouse.position = (constants.MIN_X + (x * constants.SCREEN_WIDTH),
                      constants.MIN_Y + (y * constants.SCREEN_HEIGHT))
    pass


def lower_volume():
    global volume
    if volume > 4:
        volume = volume - 5
        set_volume(volume)


def raise_volume():
    global volume
    if volume < 96:
        volume = volume + 5
        set_volume(volume)


def set_volume(volume):
    try:
        if (volume <= 100) and (volume >= 0):
            call(["amixer", "-D", "pulse", "sset", "Master", str(volume) + "%"])
            valid = True

    except ValueError as error:
        print(error)


# Main script
start_camera()
