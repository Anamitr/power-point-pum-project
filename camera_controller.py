import os

import cv2
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

import constants
import image_util
import recognition_util
from constants import PROJECT_NAME

CLASSIFIER_NAME = 'svc_classifier_3.pkl'
PCA_NAME = 'pca_3.pkl'

classifier: LinearSVC = None
pca: PCA = None


def start_camera():
    camera = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')
    i = 1

    while True:
        (grabbed, frame) = camera.read()
        cv2.imshow(PROJECT_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            # # i += 1
            #
            # image = cv2.imread(os.path.join('temp', str(i) + ".jpg"))
            black_and_white_image = image_util.get_black_and_white_hand(frame)
            cv2.imwrite(os.path.join('temp', "temp.jpg"), black_and_white_image)

            images = [black_and_white_image]
            # features = [image_util.get_features_for_one_image(black_and_white_image)]
            features = image_util.getFeatures(images)
            features_normalized = recognition_util.get_normalized_features(features)
            features_pca = pca.transform(features_normalized)
            predicted = classifier.predict(features_pca)
            print("predicted:", predicted)

        # black_and_white_img = None
        # try:
        #     black_and_white_img = image_util.show_image(image_util.get_black_and_white_hand(frame))
        # except Exception:
        #     print("Not able to extract black and white img!")
        # image_util.show_image(black_and_white_img)

        # if black_and_white_img is not None:
        #     moments = cv2.moments(black_and_white_img)
        #     hu_moments = cv2.HuMoments(moments)
        #     print(hu_moments)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Main script
classifier = recognition_util.open_model(constants.TRAINED_MODELS_FOLDER + '/' + CLASSIFIER_NAME)
pca = recognition_util.open_model(constants.TRAINED_MODELS_FOLDER + '/' + PCA_NAME)
start_camera()
