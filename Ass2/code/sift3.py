import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters



def SIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    octaves = create_octaves(gray)
    dog = get_dog(octaves)
    candidates = find_keypoint_candidates(dog)
    keypoints = refine_keypoints(candidates, octaves)
    assign_orientations(keypoints, octaves)
    descriptors,key_pts = compute_descriptors(keypoints, octaves)
    
    return keypoints, descriptors

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


def assign_orientations(keypoints, octaves):
    for octave_no,octave_keypoints in enumerate(keypoints):
        for i,keypoint in enumerate(octave_keypoints):
            x, y, scale = keypoint['x'], keypoint['y'], keypoint['scale']
            hist = compute_gradient_orientation_histogram(octaves[octave_no][scale], x, y)
            peak = np.argmax(hist)
            orientation = peak * (360 / 36)
            keypoint['orientation'] = orientation

def refine_keypoints(candidates, octaves):
    keypoints = []
    for i in range(len(octaves)):
        octave_keypoints = []
        for j,candidate in enumerate(candidates[i]):
            if j==0:
                continue 
            scale = j
            for cand in candidate:
                x, y = cand
                if is_extremum(octaves[i][scale], x, y):
                #if True :
                    #refined_x, refined_y = refine_position(octaves[i][scale], x, y)
                    refined_x, refined_y = x,y
                    if is_edge_point(octaves[i][scale] ,octaves[i][scale-1], refined_x, refined_y):
                        continue
                    if is_low_contrast(octaves[i][scale], refined_x, refined_y):
                        continue
                    #if (scale ==1 or scale==2) and (is_max_all_scale_pt(cand,octaves[i]) or is_min_all_scale_pt(cand,octaves[i])):
                    keypoint = {'x': int(refined_x), 'y': int(refined_y), 'scale': scale}
                    octave_keypoints.append(keypoint)
        keypoints.append(octave_keypoints)
    return keypoints

def is_max_all_scale_pt(cand,octave_scal_list):
    x,y =cand
    #print(x)
    mx=True
    size_y, size_x  = octave_scal_list[0].shape
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if (i>=0 and i<size_x) and (j>=0 and j<size_y) and octave_scal_list[1][y][x]< octave_scal_list[1+k][y+i][x+j]:
                    mx=False
    return mx

def is_min_all_scale_pt(cand,octave_scal_list):
    x,y =cand
    print(x)
    mx=True
    size_y, size_x  = octave_scal_list[0].shape
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                if (i>=0 and i<size_x) and (j>=0 and j<size_y) and octave_scal_list[1][y][x]> octave_scal_list[1+k][y+i][x+j]:
                    mx=False
    return mx

def is_low_contrast(octave, x, y, threshold=0.03):
    #print(f'x {x} y {y} \n')
    return np.abs(octave[int(y), int(x)]) < threshold

def is_extremum(octave, x, y):
    value = octave[y, x]
    neighbors = []
    for dx in range(-4, 5):
        for dy in range(-4, 5):
            if dx == 0 and dy == 0:
                continue
            if (x+dx<0 or x+dx> octave.shape[1]-1) or (y+dy<0 or y+dy> octave.shape[0]-1):
                continue
            neighbors.append(octave[y+dy, x+dx])
    return all(value > n for n in neighbors) or all(value < n for n in neighbors)

def is_edge_point(octave,octave2, x, y, threshold=10):

    Dx = (octave[int(y), int(x+1)] - octave[int(y), int(x-1)]) / 2
    Dy = (octave[int(y+1), int(x)] - octave[int(y-1), int(x)]) / 2
    Ds = (octave[int(y), int(x)] - octave2[int(y), int(x)]) / 2
    
    Dxx = octave[int(y), int(x+1)] - 2 * octave[int(y), int(x)] + octave[int(y), int(x-1)]
    Dyy = octave[int(y+1), int(x)] - 2 * octave[int(y), int(x)] + octave[int(y-1), int(x)]
    Dxy = (octave[int(y+1), int(x+1)] - octave[int(y+1), int(x-1)] - octave[int(y-1), int(x+1)] + octave[int(y-1), int(x-1)]) / 4
    
    tr_H = Dxx + Dyy
    det_H = Dxx * Dyy - Dxy**2
    
    curvature_ratio = tr_H**2 / det_H
    
    return det_H < 0 or curvature_ratio > (threshold + 1)**2 / threshold

def refine_position(octave, x, y, threshold=0.5):
    dx = (octave[y, x+1] - octave[y, x-1]) / 2
    dy = (octave[y+1, x] - octave[y-1, x]) / 2
    ds = (octave[y, x] - octave[y, x]) / 2
    
    dxx = (octave[y, x+1] - 2 * octave[y, x] + octave[y, x-1])
    dyy = (octave[y+1, x] - 2 * octave[y, x] + octave[y-1, x])
    dss = (octave[y, x] - 2 * octave[y, x] + octave[y, x])
    
    dxy = (octave[y+1, x+1] - octave[y+1, x-1] - octave[y-1, x+1] + octave[y-1, x-1]) / 4
    dxs = (octave[y, x+1] - octave[y, x-1] - octave[y, x+1] + octave[y, x-1]) / 4
    dys = (octave[y+1, x] - octave[y-1, x] - octave[y+1, x] + octave[y-1, x]) / 4
    
    dD = np.array([dx, dy, ds])
    H = np.array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])
    
    offset = -np.linalg.lstsq(H, dD, rcond=None)[0]
    
    if np.abs(offset[0]) < threshold and np.abs(offset[1]) < threshold and np.abs(offset[2]) < threshold:
        return x + offset[0], y + offset[1]
    else:
        return x, y
    
def find_keypoint_candidates(octaves, threshold_abs=0.01):
    candidates = []
    for octave in octaves:
        octave_candidates = []
        for j,scale in enumerate(octave):
            if(j!=0 and j!=len(octave)-1):
                print(f'{j} \n')
                local_maxima = find_local_maxima_3d(octave,j, threshold_abs)
                local_minima = find_local_minima_3d(octave,j, threshold_abs)
                octave_candidates.append(local_maxima)
                octave_candidates.append(local_minima)
        candidates.append(octave_candidates)
    return candidates

def find_local_maxima(image, threshold_abs):
    local_maxima = []
    h, w = image.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if image[y, x] >= threshold_abs:
                is_max = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if image[y, x] < image[y + dy, x + dx]:
                            is_max = False
                            break
                    if not is_max:
                        break
                if is_max:
                    local_maxima.append((x, y))
    return local_maxima

def find_local_maxima_3d(image,scale, threshold_abs):
    local_maxima = []
    h, w = image[scale].shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if image[scale][y, x] >= threshold_abs:
                is_max = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        for ds in [-1,0,1]:
                            if image[scale][y, x] < image[scale+ds][y + dy, x + dx]:
                                is_max = False
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
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        for ds in [-1,0,1]:
                            if image[scale][y, x] > image[scale+ds][y + dy, x + dx]:
                                is_max = False
                                break
                        if not is_max:
                            break
                if is_max:
                    local_maxima.append((x, y))
    return local_maxima


def create_octaves(image, num_octaves=3, num_scales=5, sigma=1.6):
    octaves = []
    k = 2**(1/num_scales)

    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

    for _ in range(num_octaves):
        octave = []
        for _ in range(num_scales):
            image = gaussian_filter(image, sigma)
            octave.append(image)
            sigma *= k
        octaves.append(octave)
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        
    return octaves

def compute_descriptors(keypoints, octaves):
    descriptors = []
    key_pts=[]
    for i,octave_keypoints in enumerate(keypoints):
        for keypoint in octave_keypoints:
            x, y, scale, orientation = keypoint['x'], keypoint['y'], keypoint['scale'], keypoint['orientation']
            descriptor = compute_descriptor(octaves[i][scale], x, y, orientation)
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
                    px = x + (i * 4) + l - 8  # Adjusting for patch size and location around (x, y)
                    py = y + (j * 4) + m - 8  # Adjusting for patch size and location around (x, y)
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

def match_descriptors(descriptors1, descriptors2):
    matches = []
    for i, desc1 in enumerate(descriptors1):
        best_match_idx = -1
        best_distance = np.inf
        second_best_distance = np.inf
        for j, desc2 in enumerate(descriptors2):
            distance = np.linalg.norm(desc1 - desc2)
            if distance < best_distance:
                second_best_distance = best_distance
                best_distance = distance
                best_match_idx = j
        if best_distance < 0.75 * second_best_distance:  # Apply distance ratio test
            matches.append((i, best_match_idx))
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

def preprocess_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_l_channel = clahe.apply(l_channel)

    equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])

    equalized_rgb_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

    return equalized_rgb_image



def main():
    img1 = cv2.imread('Images/Field/1.jpg')
    keypoints, _ = SIFT(img1)
    
    # Draw circles around keypoints
    for i,octave_keypoints in enumerate(keypoints):
        for keypoint in octave_keypoints:
            x= keypoint['x']
            y=keypoint['y']
            cv2.circle(img1, ((2**(i+1))*x, (2**(i+1))*y), 5, (0, 255, 0), 2)  # Green circles with radius 5 and thickness 2
    
    # Display the image with keypoints
    cv2.namedWindow('Image with Keypoints', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Keypoints', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()