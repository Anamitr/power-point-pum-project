import numpy as np
import os

import cv2


def rename_files():
    path = "db"

    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    print(subfolders)

    for folder in subfolders:
        files = os.listdir(folder)
        i = 1
        for file in files:
            src = folder + '\\' + file
            dst = folder + '\\' + str(i) + '.jpg'
            os.rename(src, dst)
            i = i + 1


def generate_black_and_white():
    image = cv2.imread('./db/open/2.jpg')
    h1_mask = get_image_mask(image)

    contours, hierarchy = cv2.findContours(h1_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # [print(len(contour)) for contour in contours]
    longest_contour = get_longest_contour(contours)
    print(len(longest_contour))
    new_contours = [longest_contour]

    h1_contours = np.copy(image)

    cv2.drawContours(h1_contours, new_contours, -1, (0, 0, 255), 20, hierarchy=hierarchy, maxLevel=0)

    show_image(h1_contours)

    return

    # Cropping image
    ret, thresh = cv2.threshold(h1_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    moments = cv2.moments(cnt)
    print(len(contours))

    cv2.drawContours(h1_mask, contours, -1, (0, 255, 0), 3)
    show_image(h1_mask)

    print('image: ' + str(type(h1)) + ', h1_mask: ' + str(type(h1_mask)))

    # im = h1_mask.clone()

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    h1 = cv2.drawContours(h1, [box], 0, (0, 0, 255), 2)

    # image = crop_minAreaRect(h1, rect)

    show_image(h1)


def get_longest_contour(contours):
    max_length = -1
    longest_contour = None
    for contour in contours:
        if len(contour) > max_length:
            longest_contour = contour
            max_length = len(contour)
    return longest_contour


def get_image_mask(h1):
    # zmiana przestrzeni kolorów z BGR do HSV
    h1_hsv = cv2.cvtColor(h1, cv2.COLOR_BGR2HSV)

    # określenie zakresu kolorów, które nas interesują (w przestrzeni HSV)
    lower = np.array([40, 40, 40])
    upper = np.array([70, 255, 255])

    # Progowanie obrazu za pomocą zdefiniowanych zakresów
    h1_mask = cv2.inRange(h1_hsv, lower, upper)
    # show_image(h1_mask)
    return h1_mask


def test():
    open_mask = get_image_mask()

    contours, hierarchy = cv2.findContours(open_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("num of contours: {}".format(len(contours)))

    mult = 1.2  # I wanted to show an area slightly larger than my min rectangle set this to one if you don't
    img_box = cv2.cvtColor(open_mask.copy(), cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_box, [box], 0, (0, 255, 0), 2)  # this was mostly for debugging you may omit

        W = rect[1][0]
        H = rect[1][1]

        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        rotated = False
        angle = rect[2]

        if angle < -45:
            angle += 90
            rotated = True

        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        size = (int(mult * (x2 - x1)), int(mult * (y2 - y1)))
        cv2.circle(img_box, center, 10, (0, 255, 0), -1)  # again this was mostly for debugging purposes

        M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

        cropped = cv2.getRectSubPix(img_box, size, center)
        cropped = cv2.warpAffine(cropped, M, size)

        croppedW = W if not rotated else H
        croppedH = H if not rotated else W

        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW * mult), int(croppedH * mult)),
                                           (size[0] / 2, size[1] / 2))

        show_image(croppedRotated)

    show_image(img_box)


def crop_minAreaRect(img, rect):
    # rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    return img_rot[pts[1][1]:pts[0][1],
           pts[1][0]:pts[2][0]]


def show_image(image):
    cv2.imshow("hand1", cv2.resize(image, (300, 400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()