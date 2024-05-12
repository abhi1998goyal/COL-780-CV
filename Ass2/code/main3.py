
import cv2
import numpy as np
import matplotlib.pyplot as plt



def preprocess_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_l_channel = clahe.apply(l_channel)

    equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])

    equalized_rgb_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

    return equalized_rgb_image



def cylindrical_warp(img, K):
    h, w = img.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x, map_y = map_x.astype(np.float32), map_y.astype(np.float32)

    # Unproject points from image plane to 3D space
    Xc = map_x - K[0, 2]
    Yc = map_y - K[1, 2]
    Zc = K[0, 0]

    # Project 3D points to cylindrical coordinates
    theta = np.arctan2(Xc, Zc)
    rho = Yc / np.sqrt(Xc**2 + Zc**2)

    # Map cylindrical coordinates to image coordinates
    map_x = rho * w
    map_y = (theta + np.pi) / (2 * np.pi) * h

    # Perform the remapping
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    return warped_img





def main():

    img1 = cv2.imread(f'Images/Office/1.jpg')
    img2 = cv2.imread(f'Images/Office/3.jpg')


    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)



    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]


    fx = 800  # Focal length in pixels
    fy = 800
    cx = img1.shape[1] / 2  # Principal point (image center)
    cy = img1.shape[0] / 2

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Perform cylindrical warping
    warped_img1 = cylindrical_warp(img1, K)
    warped_img2 = cylindrical_warp(img2, K)


    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', warped_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
