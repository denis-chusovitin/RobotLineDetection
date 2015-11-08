import numpy as np
import cv2
import glob

def get_camera_matrix():
    square_size = 0.3
    pattern_size = (8, 6)

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = 0, 0

    img_names = glob.glob('*.png')

    for fn in img_names:
        img = cv2.imread(fn, 0)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            img_points.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)

    _, camera_matrix, _, _, _ = cv2.calibrateCamera(obj_points, img_points, (w, h))

    return camera_matrix
