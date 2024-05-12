import cv2
import numpy as np
import matplotlib.pyplot as plt

def fast_corner_detection(image, threshold):
    mask = np.array([[1, 2, 3, 3, 3, 2, 1],
                     [2, 4, 5, 5, 5, 4, 2],
                     [3, 5, 0, 0, 0, 5, 3],
                     [3, 5, 0, 0, 0, 5, 3],
                     [3, 5, 0, 0, 0, 5, 3],
                     [2, 4, 5, 5, 5, 4, 2],
                     [1, 2, 3, 3, 3, 2, 1]], dtype=np.uint8)

    corner_response = cv2.filter2D(image, -1, mask)
    corners = []

    for y in range(3, image.shape[0] - 3):
        for x in range(3, image.shape[1] - 3):
            if corner_response[y, x] > threshold:
                is_corner = check_corners(image, x, y, corner_response[y, x])
                if is_corner:
                    corners.append((x, y))

    return corners

def check_corners(image, x_center, y_center, threshold):
    offsets = [(0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
               (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)]

    for dx, dy in offsets:
        if image[y_center + dy, x_center + dx] - image[y_center, x_center] < threshold:
            return False
    return True

def compute_brief_descriptors(image, keypoints, patch_size=31, num_pairs=256):
    descriptors = []

    for keypoint in keypoints:
        x, y = keypoint
        if x - patch_size // 2 >= 0 and x + patch_size // 2 < image.shape[1] and \
           y - patch_size // 2 >= 0 and y + patch_size // 2 < image.shape[0]:
            patch = image[y - patch_size // 2: y + patch_size // 2 + 1, x - patch_size // 2: x + patch_size // 2 + 1]
            patch = cv2.resize(patch, (patch_size, patch_size))

            # Generate random pairs of points for BRIEF descriptor
            points = np.random.randint(-patch_size // 2, patch_size // 2, size=(num_pairs, 2))
            descriptors.append(generate_brief_descriptor(patch, points))

    return descriptors


def generate_brief_descriptor(patch, points):
    descriptor = []

    for p1, p2 in points:
        p1_x, p1_y = p1 + patch.shape[1] // 2, p1 + patch.shape[0] // 2
        p2_x, p2_y = p2 + patch.shape[1] // 2, p2 + patch.shape[0] // 2

        # Make sure the points are within the patch boundaries
        if 0 <= p1_x < patch.shape[1] and 0 <= p1_y < patch.shape[0] and 0 <= p2_x < patch.shape[1] and 0 <= p2_y < patch.shape[0]:
            descriptor.append(patch[p1_y, p1_x] < patch[p2_y, p2_x])

    return np.array(descriptor, dtype=np.uint8)

def match_descriptors(descriptors1, descriptors2, threshold=70):
    matches = []

    for i, desc1 in enumerate(descriptors1):
        best_match_idx = -1
        best_distance = np.inf
        for j, desc2 in enumerate(descriptors2):
            distance = hamming_distance(desc1, desc2)
            if distance < best_distance:
                best_distance = distance
                best_match_idx = j
        if best_distance <= threshold:
            matches.append((i, best_match_idx))

    return matches

def hamming_distance(desc1, desc2):
    if len(desc1)!=len(desc2):
       return np.inf
    return np.sum(desc1 != desc2)


def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched_image

def main():
    img1 = cv2.imread('Images/Mountain/1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('Images/Mountain/2.jpg', cv2.IMREAD_GRAYSCALE)
    
    keypoints1 = fast_corner_detection(img2, threshold=100)
    keypoints2 = fast_corner_detection(img3, threshold=100)
    
    # Convert keypoints to cv2.KeyPoint objects
    keypoints1_cv2 = [cv2.KeyPoint(x, y, 10) for x, y in keypoints1]
    keypoints2_cv2 = [cv2.KeyPoint(x, y, 10) for x, y in keypoints2]
    
    descriptors1 = compute_brief_descriptors(img1, keypoints1)
    descriptors2 = compute_brief_descriptors(img2, keypoints2)
    
    matches = match_descriptors(descriptors1, descriptors2)
    
    # Draw matches using cv2.drawMatches
    matched_image = cv2.drawMatches(img1, keypoints1_cv2, img2, keypoints2_cv2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.imshow(matched_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()