import cv2
import glob
import numpy as np

def mouse_coord(event, x, y, flags, param):
    global img_points

    if event == cv2.EVENT_LBUTTONDOWN:
        img_points.append((x, y))


video_names = glob.glob('*.avi')

size = 30

board_coord = np.float32([[1, 8], [1, 1], [6, 1], [6, 8]])
board_points = size * board_coord

for v in video_names:
    img_points = []
    capture = cv2.VideoCapture(v)

    flag, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("img")
    cv2.setMouseCallback('img', mouse_coord)

    while len(img_points) < 4:
        cv2.imshow('img', gray)
        key = cv2.waitKey(1) & 0xFF

        if key == 27: # esc key
            break

    img_points = np.asarray(img_points, np.float32)
    if len(img_points) > 0:
        print cv2.findHomography(board_coord, img_points)[0]
    else:
        print None
