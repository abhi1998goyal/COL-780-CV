import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import filters

def SIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    octaves = create_octaves(gray)
    dog = get_dog(octaves)
    candidates = find_keypoint_candidates(dog)
    keypoints = refine_keypoints(candidates, octaves)
    assign_orientations(keypoints, gray)
    descriptors,key_pts = compute_descriptors(keypoints, gray)
    #return candidates
    return key_pts,descriptors




def compute_descriptors(keypoints, octaves):
    descriptors = []
    key_pts=[]
    for i,octave_keypoints in enumerate(keypoints):
        for keypoint in octave_keypoints:
            x, y, orientation = keypoint['x'], keypoint['y'],keypoint['orientation']
            descriptor = compute_descriptor(octaves, x, y, orientation)
            descriptors.append(descriptor)
            key_pts.append((x,y))
    return descriptors,key_pts


def compute_descriptor(octave, x, y, orientation, num_bins=8, patch_size=16):
    dx = filters.sobel_v(octave)
    dy = filters.sobel_h(octave)
    magnitudes = np.sqrt(dx**2 + dy**2)
    orientations = np.arctan2(dy, dx) * 180 / np.pi
    orientations[orientations < 0] += 360
    
    descriptor = []

    for i in range(0, 4):
        for j in range(0, 4):
            hist = np.zeros(num_bins)
            for l in range(0, 4):
                for m in range(0, 4):
                    px = x + (i * 4) + l - 32
                    py = y + (j * 4) + m - 32
                    if px >= 0 and px < octave.shape[1] and py >= 0 and py < octave.shape[0]:
                        relative_orientation = orientations[py, px] - orientation
                        if relative_orientation < 0:
                            relative_orientation += 360
                        if relative_orientation == 360:
                            relative_orientation =0
                        bin_idx = int(relative_orientation * num_bins / 360)
                        hist[bin_idx] += magnitudes[py, px]
            descriptor.extend(hist)
    
    # Normalize the descriptor
    descriptor = np.array(descriptor)
    descriptor /= np.linalg.norm(descriptor)
    descriptor[descriptor > 0.2] = 0.2
    descriptor /= np.linalg.norm(descriptor)
    
    return descriptor

def assign_orientations(keypoints, octaves):
    for octave_no,octave_keypoints in enumerate(keypoints):
        for i,keypoint in enumerate(octave_keypoints):
            x, y = keypoint['x'], keypoint['y']
            hist = compute_gradient_orientation_histogram(octaves, x , (2**(octave_no))*y)
            peak = np.argmax(hist)
            orientation = peak * (360 / 36)
            keypoint['orientation'] = orientation


def compute_gradient_orientation_histogram(octave, x, y, size=16, num_bins=36):
    dx = filters.sobel_v(octave)
    dy = filters.sobel_h(octave)
    magnitudes = np.sqrt(dx**2 + dy**2)
    orientations = np.arctan2(dy, dx) * 180 / np.pi
    orientations[orientations < 0] += 360
    
    hist = np.zeros(num_bins)
    
    for i in range(-size//2, size//2):
        for j in range(-size//2, size//2):
            if x+i < 0 or x+i >= octave.shape[1] or y+j < 0 or y+j >= octave.shape[0]:
                continue
            # Ensure the orientation is within the valid range [0, 360)
            orientation = orientations[y+j, x+i] % 360
            bin_idx = int(orientation * num_bins / 360)
            hist[bin_idx] += magnitudes[y+j, x+i]
    
    return hist

def refine_keypoints(candidates, octaves):
    keypoints = []
    for i in range(len(octaves)):
        octave_keypoints = []
        for j,candidate in enumerate(candidates[i]):
            #if j==0:
             #   continue 
            #scale = j
           # for cand in candidate:
                x, y = candidate
                #if is_extremum(octaves[i][scale], x, y):
                if True :
                    #refined_x, refined_y = refine_position(octaves[i][scale], x, y)
                    #refined_x, refined_y = refine_position(octaves[i][scale], octaves[i][scale-1], octaves[i][scale+1], x, y)
                    refined_x, refined_y = (2**(i))*x,(2**(i))*y
                   # if is_edge_point(octaves[i][scale] ,octaves[i][scale-1], octaves[i][scale+1] ,refined_x, refined_y):
                    #    continue
                    #if is_low_contrast(octaves[i][scale], refined_x, refined_y):
                     #   continue
                    #if (scale ==1 or scale==2) and (is_max_all_scale_pt(cand,octaves[i]) or is_min_all_scale_pt(cand,octaves[i])):
                    keypoint = {'x': int(refined_x), 'y': int(refined_y)}
                    octave_keypoints.append(keypoint)
        keypoints.append(octave_keypoints)
    return keypoints

def refine_position(octave, octave2, octave3, x, y, threshold=0.2):
    dx = (octave[y, x+1] - octave[y, x-1]) / 2
    dy = (octave[y+1, x] - octave[y-1, x]) / 2
    ds = (octave3[y, x] - octave2[y, x]) / 2
    
    dxx = (octave[y, x+1] - 2 * octave[y, x] + octave[y, x-1])
    dyy = (octave[y+1, x] - 2 * octave[y, x] + octave[y-1, x])
    dss = (octave3[y, x] - 2 * octave[y, x] + octave2[y, x])
    
    dxy = (octave[y+1, x+1] - octave[y+1, x-1] - octave[y-1, x+1] + octave[y-1, x-1]) / 4
    dxs = (octave3[y, x+1] - octave3[y, x-1] - octave2[y, x+1] + octave2[y, x-1]) / 4
    dys = (octave3[y+1, x] - octave3[y-1, x] - octave2[y+1, x] + octave2[y-1, x]) / 4
    
    dD = np.array([dx, dy, ds])
    H = np.array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])
    
    offset = -np.linalg.lstsq(H, dD, rcond=None)[0]
    
    refined_x = x + offset[0]
    refined_y = y + offset[1]
    
    if np.abs(refined_x - x) < threshold and np.abs(refined_y - y) < threshold:
        return refined_x, refined_y
    else:
        return x, y

def is_edge_point(octave,octave2, octave3,x, y, threshold=10):

    Dx = (octave[int(y), int(x+1)] - octave[int(y), int(x-1)]) / 2
    Dy = (octave[int(y+1), int(x)] - octave[int(y-1), int(x)]) / 2
    #Ds = (octave3[int(y), int(x)] - octave2[int(y), int(x)]) / 2
    
    Dxx = octave[int(y), int(x+1)] - 2 * octave[int(y), int(x)] + octave[int(y), int(x-1)]
    Dyy = octave[int(y+1), int(x)] - 2 * octave[int(y), int(x)] + octave[int(y-1), int(x)]
    Dxy = (octave[int(y+1), int(x+1)] - octave[int(y+1), int(x-1)] - octave[int(y-1), int(x+1)] + octave[int(y-1), int(x-1)]) / 4
    
    tr_H = Dxx + Dyy
    det_H = Dxx * Dyy - Dxy**2
    
    curvature_ratio = tr_H**2 / det_H
    
    return det_H < 0 or curvature_ratio > (threshold + 1)**2 / threshold

def is_low_contrast(octave, x, y, threshold=0.03):
    #print(f'x {x} y {y} \n')
    return np.abs(octave[int(y), int(x)]) < threshold

def is_extremum(octave, x, y):
    value = octave[y, x]
    neighbors = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            if (x+dx<0 or x+dx> octave.shape[1]-1) or (y+dy<0 or y+dy> octave.shape[0]-1):
                continue
            neighbors.append(octave[y+dy, x+dx])
    return all(value > n for n in neighbors) or all(value < n for n in neighbors)

def create_octaves(image, num_octaves=4, num_scales=4, sigma=1.6):
    octaves = []
    k = 2**(1/num_scales)
    #image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    for _ in range(num_octaves):
        octave = []
        for _ in range(num_scales):
            image = gaussian_filter(image, sigma)
            octave.append(image)
            sigma *= k
        octaves.append(octave)
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
    return octaves

def get_dog(octaves):
    octave_dog=[]
    for octave in octaves:
        scale_dog=[]
        for i,scale in enumerate(octave):
            if(i!=0):
                scale_dg=scale-octave[i-1]
                scale_dog.append(scale_dg)
        octave_dog.append(scale_dog)
    return octave_dog

def find_keypoint_candidates(octaves, threshold_abs=0.01):
    candidates = []
    for octave in octaves:
        octave_candidates = []
        for j,scale in enumerate(octave):
            if(j!=0 and j!=len(octave)-1):
                print(f'{j} \n')
                local_maxima = find_local_maxima_3d(octave,j, threshold_abs)
                local_minima = find_local_minima_3d(octave,j, threshold_abs)
                local_maxima.extend(local_minima)
                octave_candidates.extend(local_maxima)
                #octave_candidates.append(local_minima)
        candidates.append(octave_candidates)
    return candidates


def find_local_maxima_3d(image,scale, threshold_abs):
    local_maxima = []
    h, w = image[scale].shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if image[scale][y, x] >= threshold_abs:
                is_max = True
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        for ds in [-1,0,1]:
                            if image[scale][y, x] < image[scale+ds][y + dy, x + dx]:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if not is_max:
                            break
                if is_max:
                    local_maxima.append((x, y))
    return local_maxima

def find_local_minima_3d(image,scale, threshold_abs):
    local_maxima = []
    h, w = image[scale].shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if image[scale][y, x] >= threshold_abs:
                is_max = True
                for dy in range(-1,2):
                    for dx in range(-1,2):
                        for ds in [-1,0,1]:
                            if image[scale][y, x] > image[scale+ds][y + dy, x + dx]:
                                is_max = False
                                break
                        if not is_max:
                            break
                    if not is_max:
                            break
                if is_max:
                    local_maxima.append((x, y))
    return local_maxima

# def match_descriptors(descriptors1, descriptors2):
#     matches = []
#     for i, desc1 in enumerate(descriptors1):
#         best_match_idx = -1
#         best_distance = np.inf
#         second_best_distance = np.inf
#         for j, desc2 in enumerate(descriptors2):
#             distance = np.linalg.norm(desc1 - desc2)
#             if distance < best_distance:
#                 second_best_distance = best_distance
#                 best_distance = distance
#                 best_match_idx = j
#         if best_distance < 0.75 * second_best_distance:  # Apply distance ratio test
#             matches.append((i, best_match_idx))
#     return matches

def match_descriptors(descriptors1, descriptors2):
    distances = []
    for i, desc1 in enumerate(descriptors1):
        for j, desc2 in enumerate(descriptors2):
            distance = np.linalg.norm(desc1 - desc2)
            distances.append((i, j, distance))
    distances.sort(key=lambda x: x[2])
    matched_indices1 = set()
    matched_indices2 = set()
    matches = []
    for i, j, distance in distances:
        if len(matches) >= 10:
            break
        if i not in matched_indices1 and j not in matched_indices2:
            matches.append((i, j))
            matched_indices1.add(i)
            matched_indices2.add(j)
    return matches


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    img_matches[:h1, :w1] =img1
    img_matches[:h2, w1:w1+w2] = img2

    for idx1, idx2 in matches:
        (x1, y1) = keypoints1[idx1]
        (x2, y2) = keypoints2[idx2]
        cv2.line(img_matches, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (0, 255, 0), 1)
        cv2.circle(img_matches, (int(x1), int(y1)), 2, (0, 0, 255), -1)
        cv2.circle(img_matches, (int(x2) + w1, int(y2)), 2, (0, 0, 255), -1)
    
    return img_matches

def transform_with_homography(h_mat, points_array):
    # add column of ones so that matrix multiplication with homography matrix is possible
    ones_col = np.ones((points_array.shape[0], 1))
    points_array = np.concatenate((points_array, ones_col), axis=1)
    transformed_points = np.matmul(h_mat, points_array.T)
    epsilon = 1e-7 # very small value to use it during normalization to avoid division by zero
    transformed_points = transformed_points / (transformed_points[2,:].reshape(1,-1) + epsilon)
    transformed_points = transformed_points[0:2,:].T
    return transformed_points


def compute_outliers(h_mat, points_img_a, points_img_b, threshold=3):
    num_points = points_img_a.shape[0]
    outliers_count = 0

    # transform the match point in image B to image A using the homography
    points_img_b_hat = transform_with_homography(h_mat, points_img_b)
    
    # let x, y be coordinate representation of points in image A
    # let x_hat, y_hat be the coordinate representation of transformed points of image B with respect to image A
    x = points_img_a[:, 0]
    y = points_img_a[:, 1]
    x_hat = points_img_b_hat[:, 0]
    y_hat = points_img_b_hat[:, 1]
    euclid_dis = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in euclid_dis:
        if dis > threshold:
            outliers_count += 1
    return outliers_count

def calculate_homography(points_img_a, points_img_b):

    # concatenate the two numpy points array to get 4 columns (u, v, x, y)
    points_a_and_b = np.concatenate((points_img_a, points_img_b), axis=1)
    A = []
    # fill the A matrix by looping through each row of points_a_and_b containing u, v, x, y
    # each row in the points_ab would fill two rows in the A matrix
    for u, v, x, y in points_a_and_b:
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    
    A = np.array(A)
    _, _, v_t = np.linalg.svd(A)

    # soltion is the last column of v which means the last row of its transpose v_t
    h_mat = v_t[-1, :].reshape(3,3)
    return h_mat


def compute_homography_ransac(matches_a, matches_b):
    num_all_matches =  matches_a.shape[0]
    # RANSAC parameters
    SAMPLE_SIZE = 5 #number of point correspondances for estimation of Homgraphy
    SUCCESS_PROB = 0.995 #required probabilty of finding H with all samples being inliners 
    min_iterations = int(np.log(1.0 - SUCCESS_PROB)/np.log(1 - 0.5**SAMPLE_SIZE))
    
    # Let the initial error be large i.e consider all matched points as outliers
    lowest_outliers_count = num_all_matches
    best_h_mat = None
    best_i = 0 # just to know in which iteration the best h_mat was found

    for i in range(min_iterations):
        rand_ind = np.random.permutation(range(num_all_matches))[:SAMPLE_SIZE]
        h_mat = calculate_homography(matches_a[rand_ind], matches_b[rand_ind])
        outliers_count = compute_outliers(h_mat, matches_a, matches_b)
        if outliers_count < lowest_outliers_count:
            best_h_mat = h_mat
            lowest_outliers_count = outliers_count
            best_i = i
    best_confidence_obtained = int(100 - (100 * lowest_outliers_count / num_all_matches))

    return best_h_mat


def pb1(img1 , img2):
    img1 = img1.astype('float32')
    img2 = img2.astype('float32')
    keypoints1, descriptors1 = SIFT(img1)
    keypoints2, descriptors2 = SIFT(img2)

    # Sift = cv2.SIFT_create()
    # keypoints1, descriptors1 = Sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    # keypoints2, descriptors2 = Sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

    matches = match_descriptors(descriptors1, descriptors2)
    
    #drawn_matches = draw_matches(img1, keypoints1, img2, keypoints2, matches)
    matcha=[]
    matchb=[]

    for a,b in matches:
          #matcha.append(keypoints1[a].pt)
          matcha.append(keypoints1[a])
          #matchb.append(keypoints2[b].pt)
          matchb.append(keypoints2[b])

    #return matches,keypoints1,keypoints2

    return compute_homography_ransac(np.array(matcha),np.array(matchb))


def main():
    img = cv2.imread('Images/Office/5.jpg')
    img1 = img.astype('float32')

    img2 = cv2.imread('Images/Office/4.jpg')
    img2 = img2.astype('float32')

    keypoints1, descriptors1 = SIFT(img1)
    keypoints2, descriptors2 = SIFT(img2)

    matches = match_descriptors(descriptors1, descriptors2)
    
    drawn_matches = draw_matches(img1, keypoints1, img2, keypoints2, matches)


    # for i,octave_keypoints in enumerate(keypoints):
    #     for keypoint in octave_keypoints:
    #         x, y = int(keypoint['x']), int(keypoint['y'])
    #         cv2.circle(img, (x, y), 5, (0, 255, 0), 2) 

    # cv2.imshow('Image with Keypoints', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Matches', drawn_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()