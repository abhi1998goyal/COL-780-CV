import os
import cv2
import math
import numpy as np
import sift
import os
import sys

def ReadImage(ImageFolderPath):
    Images = []
    if os.path.isdir(ImageFolderPath):
        ImageNames = os.listdir(ImageFolderPath)
        ImageNames_Split = [[int(os.path.splitext(os.path.basename(ImageName))[0]), ImageName] for ImageName in ImageNames]
        ImageNames_Split = sorted(ImageNames_Split, key=lambda x:x[0])
        ImageNames_Sorted = [ImageNames_Split[i][1] for i in range(len(ImageNames_Split))]
        
        for i in range(len(ImageNames_Sorted)):
            ImageName = ImageNames_Sorted[i]
            InputImage = cv2.imread(ImageFolderPath + "/" + ImageName)
        
            if InputImage is None:
                print("Not able to read image: {}".format(ImageName))
                exit(0)

            Images.append(InputImage)
            
    else: 
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)
    
    return Images

    
def FindMatches(BaseImage, SecImage):
    Sift = cv2.SIFT_create()
    BaseImage_kp, BaseImage_des = Sift.detectAndCompute(cv2.cvtColor(BaseImage, cv2.COLOR_BGR2GRAY), None)
    SecImage_kp, SecImage_des = Sift.detectAndCompute(cv2.cvtColor(SecImage, cv2.COLOR_BGR2GRAY), None)

    BF_Matcher = cv2.BFMatcher()
    InitialMatches = BF_Matcher.knnMatch(BaseImage_des, SecImage_des, k=2)

    GoodMatches = []
    for m, n in InitialMatches:
        if m.distance < 0.75 * n.distance:
            GoodMatches.append([m])

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

    Matches, BaseImage_kp, SecImage_kp = FindMatches(BaseImage, SecImage_Cyl)
    
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


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 main.py 1 <img dir> <output path>")
        return

    option = int(sys.argv[1])
    img_dir = sys.argv[2]
    output_path = sys.argv[3]

    if option != 1:
        print("Invalid option. Please provide option 1.")
        return

    Images = ReadImage(img_dir)
    
    baseImage, _, _ = ProjectOnCylinder(Images[0])
    for i in range(1, len(Images)):
        ci = StitchImages(baseImage, Images[i])
        baseImage = ci.copy()

    cv2.imwrite(output_path, baseImage)

if __name__ == "__main__":
    main()