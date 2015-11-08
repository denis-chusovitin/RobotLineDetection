import cv2
import glob

video_names = glob.glob('*.avi')

for v in video_names:
    capture = cv2.VideoCapture(v)

    flag, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (8, 6), cv2.cv.CV_CALIB_CB_ADAPTIVE_THRESH + cv2.cv.CV_CALIB_CB_NORMALIZE_IMAGE)

    print found