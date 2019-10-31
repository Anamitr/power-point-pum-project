import cv2
import image_util


def start_camera():
    camera = cv2.VideoCapture('https://10.187.162.29:8080/video')

    while True:
        (grabbed, frame) = camera.read()
        image_util.show_image(frame)

        black_and_white_img = None
        try:
            black_and_white_img = image_util.show_image(image_util.get_black_and_white_hand(frame))
        except Exception:
            print("Not able to extract black and white img!")
        image_util.show_image(black_and_white_img)

        if black_and_white_img is not None:
            moments = cv2.moments(black_and_white_img)
            hu_moments = cv2.HuMoments(moments)
            print(hu_moments)
