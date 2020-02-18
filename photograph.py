import os

import cv2

import image_util
from constants import BASE_IMAGE_EXTENSION

print("Running photograph")
BLACK_AND_WHITE_MODE = True


def take_photos(save_path: str):
    camera = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')
    i = 1

    while True:
        (grabbed, frame) = camera.read()
        # image_util.show_image(frame)
        if BLACK_AND_WHITE_MODE:
            baw_img = image_util.get_black_and_white_hand(frame)
            cv2.imshow('Black and white hand', baw_img)
        else:
            cv2.imshow('Photo', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            status = cv2.imwrite(os.path.join(save_path, str(i) + BASE_IMAGE_EXTENSION), frame)
            # image_util.show_image(image_util.get_black_and_white_hand(frame))
            print("Photo", i, "taken:", status)
            i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


db_path = "db3"
sign_path = "volume_down"
take_photos(db_path + "/" + sign_path + "/")
