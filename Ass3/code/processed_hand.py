import cv2
import numpy as np
import cropped_rectangle

image = cv2.imread('1.jpg')
#image2 = cv2.resize(cv2.imread('Final_dataset/open1/train/image_1.jpg'),(image.shape[1]//2,image.shape[0]//2))
image2 = cv2.imread('Final_dataset/open1/train/image_1.jpg')
image2,(yg,xg)=cropped_rectangle.get_green_window(image2)
image2=cv2.resize(image2,(120,258))

def overlay_image(background, overlay):
    bg_height, bg_width = background.shape[:2]
    ov_height, ov_width = overlay.shape[:2]

    x = np.random.randint(0, bg_width - ov_width)
    y = np.random.randint(0, bg_height - ov_height)

    background[y:y+ov_height, x:x+ov_width] = overlay

    x1 = np.random.randint(0, bg_width - ov_width//2)
    y1 = np.random.randint(0, bg_height - ov_height//2)

    background[y1:y1+ov_height//2, x1:x1+ov_width//2] = cv2.resize(overlay,(ov_width//2,ov_height//2))

    return background

result_image = overlay_image(image, image2)

cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('result_image.jpg', result_image)

