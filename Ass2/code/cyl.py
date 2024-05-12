import cv2
import numpy as np

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

def detect_and_match_keypoints(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts

def estimate_homography(src_pts, dst_pts):
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def main():
    img1 = cv2.imread('Images/Office/1.jpg')
    img2 = cv2.imread('Images/Office/3.jpg')

    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    # Define camera intrinsics (focal length and principal point)
    fx = 1000  # Focal length in pixels
    fy = 1000
    cx = img1.shape[1] / 2  # Principal point (image center)
    cy = img1.shape[0] / 2

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # Perform cylindrical warping
    warped_img1 = cylindrical_warp(img1, K)
    warped_img2 = cylindrical_warp(img2, K)

    # Detect and match keypoints
    src_pts, dst_pts = detect_and_match_keypoints(warped_img1, warped_img2)

    # Estimate homography
    H = estimate_homography(src_pts, dst_pts)

    # Warp and stitch images
    h1, w1 = warped_img1.shape[:2]
    h2, w2 = warped_img2.shape[:2]

    panorama_width = max(w1, w2)
    panorama_height = h1 + h2

    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)
    panorama[:h1, :w1] = warped_img1
    panorama[:h2, :w2] = warped_img2

    # Apply homography transformation
    for y in range(h2):
        for x in range(w2):
            point_homogeneous = np.dot(H, np.array([x, y, 1]))
            point_homogeneous /= point_homogeneous[2]
            px, py = int(point_homogeneous[0]), int(point_homogeneous[1])
            if 0 <= px < w1 and 0 <= py < h1:
                panorama[py, px] = warped_img2[y, x]

    # Display result
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
