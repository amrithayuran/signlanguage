import cv2
import config

def func(path):    
    frame = cv2.imread(path)
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, config.MIN_VALUE, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res


