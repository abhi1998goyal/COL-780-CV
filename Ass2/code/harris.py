import numpy as np
import cv2

def harris_corner_detection(image, threshold=0.01):
    # Compute derivatives using Sobel operator
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of derivatives
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Compute sums of products of derivatives using Gaussian filter
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)

    # Compute determinant and trace of the matrix M
    det_M = Sxx * Syy - Sxy ** 2
    trace_M = Sxx + Syy

    # Compute Harris response
    R = det_M - 0.04 * trace_M ** 2

    # Thresholding
    R[R < threshold * np.max(R)] = 0

    # Non-maximum suppression
    R = cv2.dilate(R, None)
    R[image == 0] = 0

    # Find keypoints
    keypoints = np.argwhere(R != 0)
    return keypoints

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2, ratio_threshold=0.7):
    matches = []

    for i, descriptor1 in enumerate(descriptors1):
        best_match_index = -1
        best_match_distance = np.inf

        for j, descriptor2 in enumerate(descriptors2):
            distance = euclidean_distance(descriptor1, descriptor2)
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_index = j

        # Ratio test
        second_best_distance = np.inf
        for j, descriptor2 in enumerate(descriptors2):
            distance = euclidean_distance(descriptor1, descriptor2)
            if j != best_match_index and distance < second_best_distance:
                second_best_distance = distance

        if best_match_distance < ratio_threshold * second_best_distance:
            matches.append((keypoints1[i], keypoints2[best_match_index]))

    return matches

def main():
    # Read images
    image1 = cv2.imread('Images/Field/1.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('Images/Field/2.jpg', cv2.IMREAD_GRAYSCALE)

    image1 = cv2.resize(image1, (image1.shape[1]//4, image1.shape[0]//4))
    image2 = cv2.resize(image1, (image1.shape[1]//4, image1.shape[0]//4))

    # Detect keypoints using Harris corner detection
    keypoints1 = harris_corner_detection(image1)
    keypoints2 = harris_corner_detection(image2)

    

    # Extract descriptors (for simplicity, let's just use pixel intensities as descriptors)
    descriptors1 = [image1[keypoint[0], keypoint[1]] for keypoint in keypoints1]
    descriptors2 = [image2[keypoint[0], keypoint[1]] for keypoint in keypoints2]

    # Match keypoints
    matches = match_keypoints(keypoints1, keypoints2, descriptors1, descriptors2)

    # Draw matches
    matched_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None)

    # Display result
    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
