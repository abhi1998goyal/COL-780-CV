import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage import filters
import os
import cv2
import math
import numpy as np
import sift



def SIFT(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    octaves = create_octaves(gray)
    candidates = find_keypoint_candidates(octaves)
    keypoints = refine_keypoints(candidates, octaves)
    
    assign_orientations(keypoints, octaves)
    descriptors,key_pts = compute_descriptors(keypoints, octaves)
    
    return key_pts, descriptors

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
            scale = j
            if j==0 or j==len(candidates[i])-1:
                continue 
            for cand in candidate:
                x, y = cand
                if is_extremum(octaves[i][scale], x, y):
                   # refined_x, refined_y = refine_position(octaves[i][scale], octaves[i][scale-1], octaves[i][scale+1], x, y)
                    refined_x, refined_y = x,y
                    #if is_edge_point(octaves[i][scale], refined_x, refined_y):
                     #   continue
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

def refine_position(octave, octave2 ,octave3, x, y, threshold=0.5):
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
    
    if np.abs(offset[0]) < threshold and np.abs(offset[1]) < threshold and np.abs(offset[2]) < threshold:
        return x + offset[0], y + offset[1]
    else:
        return x, y
    
def find_keypoint_candidates(octaves, threshold_abs=0.01):
    candidates = []
    for octave in octaves:
        octave_candidates = []
        for scale in octave:
            local_maxima = find_local_maxima(scale, threshold_abs)
            octave_candidates.append(local_maxima)
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



def create_octaves(image, num_octaves=2, num_scales=3, sigma=1.6):
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

def ReadImage(ImageFolderPath):
    Images = []									# Input Images will be stored in this list.

	# Checking if path is of folder.
    if os.path.isdir(ImageFolderPath):                              # If path is of a folder contaning images.
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x:x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
        
        for i in range(len(ImageNames_Sorted)):                     # Getting all image's name present inside the folder.
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)  # Reading images one by one.
            
            # Checking if image is read
            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                exit(0)

            Images.append(InputImage)                               # Storing images.
            
    else:                                       # If it is not folder(Invalid Path).
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)
    
    return Images

def FindHomography(matches, BaseImage_kp, SecImage_kp):
    BaseImage_pts = np.float32([BaseImage_kp[idx] for idx, _ in matches])
    SecImage_pts = np.float32([SecImage_kp[idx] for _, idx in matches])

    HomographyMatrix, _ = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    return HomographyMatrix


def GetNewFrame(hm, Sec_Img, Base_Img):

    (Height, Width) = Sec_Img
    im = np.array([[0, Width - 1, Width - 1, 0],
                              [0, 0, Height - 1, Height - 1],
                              [1, 1, 1, 1]])

    fm = np.dot(hm, im)

    [x, y, c] = fm
    x = np.divide(x, c)
    y = np.divide(y, c)

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    New_Width = max_x
    New_Height = max_y
    Correction = [0, 0]
    if min_x < 0:
        New_Width -= min_x
        Correction[0] = abs(min_x)
    if min_y < 0:
        New_Height -= min_y
        Correction[1] = abs(min_y)
    if New_Width < Base_Img[1] + Correction[0]:
        New_Width = Base_Img[1] + Correction[0]
    if New_Height < Base_Img[0] + Correction[1]:
        New_Height = Base_Img[0] + Correction[1]

    x = np.add(x, Correction[0])
    y = np.add(y, Correction[1])
    oldpts = np.float32([[0, 0],
                                   [Width - 1, 0],
                                   [Width - 1, Height - 1],
                                   [0, Height - 1]])
    newpts = np.float32(np.array([x, y]).transpose())

    hm = cv2.getPerspectiveTransform(oldpts, newpts)
    
    return [New_Height, New_Width], Correction, hm



def StitchImages(BaseImage, SecImage):
    SecImage_Cyl, mask_x, mask_y = ProjectOnCylinder(SecImage)
    SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
    SecImage_Mask[mask_y, mask_x, :] = 255

    BaseImage_kp, descriptors1 = SIFT(BaseImage)
    SecImage_kp, descriptors2 = SIFT(SecImage)

    Matches = match_descriptors(descriptors1, descriptors2)
    
    hm = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    
    NewFrameSize, Correction, hm = GetNewFrame(hm, SecImage_Cyl.shape[:2], BaseImage.shape[:2])
    SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, hm, (NewFrameSize[1], NewFrameSize[0]))
    SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, hm, (NewFrameSize[1], NewFrameSize[0]))
    BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

    StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

    return StitchedImage


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


if __name__ == "__main__":
    Images = ReadImage("Images/Field")
    
    baseImage, _, _ = ProjectOnCylinder(Images[0])
    for i in range(1, len(Images)):
        # if(i==2):
        #   continue
        #print(f'{i} \n')
        ci = StitchImages(baseImage, Images[i])

        baseImage = ci.copy()
    #img1 = ProjectOnCylinder(cv2.imread('Images/Office/2.jpg'))

    # keypoints1, descriptors1 = SIFT(ci)
    # keypoints2, descriptors2 = SIFT(img1)
    
    # matches = match_descriptors(descriptors1, descriptors2)
    # img_matches = draw_matches(ci, keypoints1, img1, keypoints2, matches)
    
    # cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    # cv2.imshow('Matches', img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("Stitched_Panorama.png", baseImage)