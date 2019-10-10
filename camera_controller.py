import cv2
import image_util


def start_camera():
    camera = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = camera.read()
        black_and_white_img = image_util.show_image(image_util.get_black_and_white_hand(frame))

        moments = cv2.moments(black_and_white_img)
        hu_moments = cv2.HuMoments(moments)
        print(hu_moments)
