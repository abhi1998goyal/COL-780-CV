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

def estimate_homography(src_pts, dst_pts):
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H


def estimate_rotational_homography(src_pts, dst_pts):
    # Calculate the centroids of source and destination key points
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    # Calculate the rotation angle
    angle_rad = np.arctan2(dst_centroid[1] - src_centroid[1], dst_centroid[0] - src_centroid[0])
    angle_deg = np.degrees(angle_rad)

    # Construct the rotational homography matrix
    H = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                  [np.sin(angle_rad), np.cos(angle_rad), 0],
                  [0, 0, 1]])

    return H

# def warp_and_stitch_images(img1, img2, H):
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]

#     warped_img2 = cv2.warpPerspective(img2, H, ( w2, h1))

#     panorama = np.zeros((h1, w1 + w2, 3), dtype=np.uint8)

#     panorama[:, :w1] = img1
#     panorama[:, w1:] = warped_img2

#     return panorama


def warp_and_stitch_images(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the inverse of the homography matrix
    H_inv = np.linalg.inv(H)

    # Warp the corner points of img2 using the inverse of the homography matrix
    corners = np.array([[0, 0, 1], [0, h2 - 1, 1], [w2 - 1, 0, 1], [w2 - 1, h2 - 1, 1]])
    corners_warped = np.dot(H_inv, corners.T).T
    corners_warped /= corners_warped[:, 2, None]

    # Calculate the maximum and minimum coordinates of the warped corner points
    min_x = int(np.floor(np.min(corners_warped[:, 0])))
    max_x = int(np.ceil(np.max(corners_warped[:, 0])))
    min_y = int(np.floor(np.min(corners_warped[:, 1])))
    max_y = int(np.ceil(np.max(corners_warped[:, 1])))

    # Calculate the size of the panorama canvas
    panorama_width = max(w1, max_x)
    panorama_height = max(h1, max_y)

    # Create an empty canvas for the panorama
    panorama = np.zeros((panorama_height, panorama_width, 3), dtype=np.uint8)

    # Paste img1 onto the panorama canvas
    panorama[:h1, :w1] = img1

    # Iterate over each pixel in img2
    for y in range(h2):
        for x in range(w2):
            # Apply inverse homography to get corresponding point in img1
            point_homogeneous = np.dot(H_inv, np.array([x, y, 1]))
            point_homogeneous /= point_homogeneous[2]
            px, py = int(point_homogeneous[0]), int(point_homogeneous[1])

            # Copy pixel value from img2 to panorama if it falls within the canvas bounds
            if w1 <= px < panorama_width and 0 <= py < panorama_height:
                panorama[py, px ] = img2[y, x]

    return panorama




def detect_and_match_keypoints(img1, img2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    img1_with_keypoints = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_keypoints = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(len(good_matches))

    matched_img = cv2.drawMatches(img1_with_keypoints, kp1, img2_with_keypoints, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.namedWindow('matched_img', cv2.WINDOW_NORMAL)
    cv2.imshow('matched_img', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return src_pts, dst_pts
   # return src_pts, dst_pts, matched_img


def Convert_xy(x, y):
    global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return xt, yt


def ProjectOnCylinder(im):
    global w, h, center,  f
    h, w = im.shape[:2]
    center = [w // 2, h // 2]
    f = 1100

    ti = np.zeros(im.shape, dtype=np.uint8)

    cord_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = cord_ti[:, 0]
    ti_y = cord_ti[:, 1]
    
    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    withinIndex = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

    ti_x = ti_x[withinIndex]
    ti_y = ti_y[withinIndex]
    
    ii_x = ii_x[withinIndex]
    ii_y = ii_y[withinIndex]

    ii_tl_x = ii_tl_x[withinIndex]
    ii_tl_y = ii_tl_y[withinIndex]

    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)
    
    ti[ti_y, ti_x, :] = ( weight_tl[:, None] * im[ii_tl_y,     ii_tl_x,     :] ) + \
                                      ( weight_tr[:, None] * im[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                      ( weight_bl[:, None] * im[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                      ( weight_br[:, None] * im[ii_tl_y + 1, ii_tl_x + 1, :] )


    min_x = min(ti_x)

    ti = ti[:, min_x : -min_x, :]

    return ti, ti_x-min_x, ti_y



def main():

    img1 = cv2.imread(f'Images/Field/1.jpg')
    img2 = cv2.imread(f'Images/Field/3.jpg')


    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)


    img1,_,_ = ProjectOnCylinder(img1)
    img2,_,_ = ProjectOnCylinder(img2)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    src_pts, dst_pts = detect_and_match_keypoints(img1, img2)
    H = estimate_homography(src_pts, dst_pts)

    #H = estimate_rotational_homography(src_pts, dst_pts)

    #H_inv = np.linalg.inv(H)


    wi=warp_and_stitch_images(img1, img2, H)
    #warped_img1 = cv2.warpPerspective(img2, H, (w2, h2))

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #panorama = warp_and_stitch_images(img1, img2, H)

    # Display result
    cv2.imshow('img', wi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
