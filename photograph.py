import os

import cv2

import image_util

print("Running photograph")


def take_photos(save_path: str):
    camera = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed')
    i = 1

    while True:
        (grabbed, frame) = camera.read()
        # image_util.show_image(frame)

        cv2.imshow('Photo', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            status = cv2.imwrite(os.path.join(save_path, str(i) + ".jpg"), frame)
            # image_util.show_image(image_util.get_black_and_white_hand(frame))
            print("Photo", i, "taken:", status)
            i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


db_path = "db3"
sign_path = "play"
take_photos(db_path + "/" + sign_path + "/")
