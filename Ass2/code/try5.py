import os
import cv2
import math
import numpy as np
import sift6

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

            Images.append(cv2.resize(InputImage, (InputImage.shape[1], InputImage.shape[0])))                               # Storing images.
            #Images.append(preprocess_image(InputImage))
    else:                                       # If it is not folder(Invalid Path).
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)
    
    return Images

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

def FindMatches(BaseImage, SecImage):
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

    #matches_a, matches_b = sift6.get_matches(BaseImage, SecImage_Cyl)

    #BaseImage_kp, BaseImage_des = sift6.computeKeypointsAndDescriptors(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY))
    #SecImage_kp, SecImage_des = sift6.computeKeypointsAndDescriptors(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY))
    
    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])
    #GoodMatches=match_descriptors(BaseImage_des,SecImage_des)

    return GoodMatches, BaseImage_kp, SecImage_kp


def FindHomography(Matches, BaseImage_kp, SecImage_kp):
    BaseImage_pts = []
    SecImage_pts = []
    for Match in Matches:
        BaseImage_pts.append(BaseImage_kp[Match[0].queryIdx].pt)
        SecImage_pts.append(SecImage_kp[Match[0].trainIdx].pt)

   
    BaseImage_pts = np.float32(BaseImage_pts)
    SecImage_pts = np.float32(SecImage_pts)

   
    (HomographyMatrix, Status) = cv2.findHomography(SecImage_pts, BaseImage_pts, cv2.RANSAC, 4.0)

    #hma = sift6.compute_homography_ransac(SecImage_pts, BaseImage_pts)

    return HomographyMatrix
    #return hma

    
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



def StitchImages(hma,BaseImage,SecImage):
    SecImage_Cyl, mask_x, mask_y = ProjectOnCylinder(1100,SecImage)
    SecImage_Mask = np.zeros(SecImage_Cyl.shape, dtype=np.uint8)
    SecImage_Mask[mask_y, mask_x, :] = 255

    #Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage_Cyl)
    #base_image_resized = cv2.resize(BaseImage, (BaseImage.shape[1], BaseImage.shape[0]))
    #SecImage_Cyl_resized = cv2.resize(SecImage_Cyl, (SecImage_Cyl.shape[1], SecImage_Cyl.shape[0]))
    #matches_a, matches_b = sift6.get_matches(BaseImage, SecImage_Cyl)
   # hma = sift6.compute_homography_ransac(matches_a, matches_b)
    
    #hm = FindHomography(Matches, BaseImage_kp, SecImage_kp)
    
    NewFrameSize, Correction, hm = GetNewFrame(hma, SecImage_Cyl.shape[:2], BaseImage.shape[:2])
    SecImage_Transformed = cv2.warpPerspective(SecImage_Cyl, hm, (NewFrameSize[1], NewFrameSize[0]))
    SecImage_Transformed_Mask = cv2.warpPerspective(SecImage_Mask, hm, (NewFrameSize[1], NewFrameSize[0]))
    BaseImage_Transformed = np.zeros((NewFrameSize[0], NewFrameSize[1], 3), dtype=np.uint8)
    BaseImage_Transformed[Correction[1]:Correction[1]+BaseImage.shape[0], Correction[0]:Correction[0]+BaseImage.shape[1]] = BaseImage

    StitchedImage = cv2.bitwise_or(SecImage_Transformed, cv2.bitwise_and(BaseImage_Transformed, cv2.bitwise_not(SecImage_Transformed_Mask)))

    return StitchedImage


def Convert_xy(x, y,center,f):
    #global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return xt, yt


def ProjectOnCylinder(fo,im):
    #global w, h, center,  f
    h, w = im.shape[:2]
    center = [w // 2, h // 2]
    f =fo

    ti = np.zeros(im.shape, dtype=np.uint8)

    cord_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = cord_ti[:, 0]
    ti_y = cord_ti[:, 1]
    
    ii_x, ii_y = Convert_xy(ti_x, ti_y,center,f)

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


def preprocess_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_l_channel = clahe.apply(l_channel)

    equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])

    equalized_rgb_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

    return equalized_rgb_image


if __name__ == "__main__":
    Images = ReadImage("Images/Mountain")
    #Images[0]=cv2.resize(Images[0], (Images[0].shape[1]//2, Images[0].shape[0]//2))
   
    baseImage, _, _ = ProjectOnCylinder(1100,Images[0])
    
    homo=[]
    for i in range(1, len(Images)):
        img1=cv2.resize(Images[i-1], (Images[i-1].shape[1], Images[i].shape[0]))
        img2=cv2.resize(Images[i], (Images[i].shape[1], Images[i].shape[0]))
        #img1, _, _ = ProjectOnCylinder(1100,img1)
        #img2, _, _ = ProjectOnCylinder(1100,img2)
        Sift = cv2.SIFT_create()
        img1_kp, img1_des =  Sift.detectAndCompute((cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)),None)
        img2_kp, img2_des =  Sift.detectAndCompute((cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)),None)
        #matches_a, matches_b = sift6.get_matches(img2, img2)
        BF_Matcher = cv2.BFMatcher()
        InitialMatches = BF_Matcher.knnMatch(img1_des, img2_des, k=2)

        GoodMatches = []
        for m, n in InitialMatches:
           if m.distance < 0.75 * n.distance:
              GoodMatches.append([m])

        hma=FindHomography(GoodMatches, img1_kp, img2_kp)
        #hma = sift6.compute_homography_ransac(matches_a, matches_b)
        homo.append(hma)
    

    for i in range(1, len(Images)):
        #print(i)

        ci = StitchImages(homo[i-1],baseImage,Images[i])

        baseImage = ci.copy()

        # cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
        # cv2.imshow('Matches',baseImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    cv2.imwrite("Stitched_Panorama.png", baseImage)