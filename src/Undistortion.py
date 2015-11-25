import cv2

def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst
