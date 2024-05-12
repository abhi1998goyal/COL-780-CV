import cv2
import numpy as np

# load image
#img = cv2.imread("Final_dataset/open1/train/image_0.jpg")
#img = cv2.imread('result_image.jpg')
#img = cv2.resize(img,(300,500))

def dec_bbox(bbox, scale_factor):
    x, y, w, h = bbox
    delta_w = int((1-scale_factor) * w / 2)
    delta_h = int((1-scale_factor) * h / 2)
    return x + delta_w, y + delta_h, w - 2 * delta_w, h - 2 * delta_h

def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_only = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(green_only, cv2.COLOR_BGR2GRAY)
    return gray

def get_green_window(img):
    threshold=preprocess_image(img)
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    
    x,y,w,h = dec_bbox(cv2.boundingRect(big_contour),0.9)
   # print(x,y,w,h)
    
    crop = img[y:y+h, x:x+w]
    return crop,(y,x)

#crop=get_green_window(img)
#cv2.imwrite("screen_threshold.jpg", threshold)
#cv2.imwrite("screen_cropped.jpg", crop)

#cv2.imshow("threshold", threshold)
#cv2.imshow("crop", crop)
#cv2.waitKey(0)