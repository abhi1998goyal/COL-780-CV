import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from skimage.feature import peak_local_max
from skimage import filters

def SIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    octaves = create_octaves(gray)
    candidates = find_keypoint_candidates(octaves)
    keypoints = refine_keypoints(candidates, octaves)


def refine_keypoints(candidates, octaves):
    keypoints = {}
    for octave_no, octave_candidates in candidates.items():
        octave_keypoints = []
        for scale_no, scale_candidates in octave_candidates.items():
            for candidate in scale_candidates:
                x, y = candidate
                if is_extremum(octaves[octave_no][scale_no], x, y):
                    refined_x, refined_y = refine_position(octaves[octave_no][scale_no], x, y)
                    if is_edge_point(octaves[octave_no][scale_no], refined_x, refined_y):
                        continue
                    if is_low_contrast(octaves[octave_no][scale_no], refined_x, refined_y):
                        continue
                    keypoint = {'x': int(refined_x), 'y': int(refined_y), 'scale': scale_no}
                    octave_keypoints.append(keypoint)
        keypoints[octave_no] = octave_keypoints
    return keypoints



def is_low_contrast(octave, x, y, threshold=0.03):
    #print(f'x {x} y {y} \n')
    return np.abs(octave[int(y), int(x)]) < threshold


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
    
def is_extremum(octave, x, y):
    value = octave[y, x]
    neighbors = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            neighbors.append(octave[y+dy, x+dx])
    return all(value > n for n in neighbors) or all(value < n for n in neighbors)

def is_edge_point(octave, x, y, threshold=10):

    Dx = (octave[int(y), int(x+1)] - octave[int(y), int(x-1)]) / 2
    Dy = (octave[int(y+1), int(x)] - octave[int(y-1), int(x)]) / 2
    Ds = (octave[int(y), int(x)] - octave[int(y), int(x)]) / 2
    
    Dxx = octave[int(y), int(x+1)] - 2 * octave[int(y), int(x)] + octave[int(y), int(x-1)]
    Dyy = octave[int(y+1), int(x)] - 2 * octave[int(y), int(x)] + octave[int(y-1), int(x)]
    Dxy = (octave[int(y+1), int(x+1)] - octave[int(y+1), int(x-1)] - octave[int(y-1), int(x+1)] + octave[int(y-1), int(x-1)]) / 4
    
    tr_H = Dxx + Dyy
    det_H = Dxx * Dyy - Dxy**2
    
    curvature_ratio = tr_H**2 / det_H
    
    return det_H < 0 or curvature_ratio > (threshold + 1)**2 / threshold


def find_keypoint_candidates(octaves, threshold_abs=70):
    keypoints = {}
    for octave_no, octave in octaves.items():
        octave_keypoints = {}
        for scale_no, scale_image in octave.items():
            local_maxima = find_local_maxima(scale_image, threshold_abs)
            octave_keypoints[scale_no] = local_maxima
        keypoints[octave_no] = octave_keypoints
    return keypoints

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

def create_octaves(image, num_octaves=4, num_scales=3, sigma=1.6):
    octaves = {}
    k = 2**(1/num_scales)
    
    for octave_no in range(num_octaves):
        octave = {}
        for scale_no in range(num_scales):
            octave[scale_no] = image
            image = gaussian_filter(image, sigma)
            sigma *= k
        octaves[octave_no] = octave
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
        
    return octaves




def main():
    img1 = cv2.imread('Images/Field/1.jpg')
    keypoints, _ = SIFT(img1)
    
    # Draw circles around keypoints
    for octave_keypoints in keypoints:
        for keypoint in octave_keypoints:
            x, y = int(keypoint['x']), int(keypoint['y'])
            cv2.circle(img1, (x, y), 5, (0, 255, 0), 2)  # Green circles with radius 5 and thickness 2
    
    # Display the image with keypoints
    cv2.imshow('Image with Keypoints', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()