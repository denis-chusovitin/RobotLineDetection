import numpy as np
import cv2

points_img = np.float32([[3, 218], [87, 150], [263, 155], [290, 227]])
points_obj = np.float32([[250, 608], [250, 500], [358, 500], [358, 608]])

h, _ = cv2.findHomography(points_img, points_obj)

print h