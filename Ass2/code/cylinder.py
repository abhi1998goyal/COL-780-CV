import os
import cv2
import math
import numpy as np

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

            Images.append(cv2.resize(InputImage, (InputImage.shape[1]//4, InputImage.shape[0]//4)))                               # Storing images.
            
    else:                                       # If it is not folder(Invalid Path).
        print("\nEnter valid Image Folder Path.\n")
        
    if len(Images) < 2:
        print("\nNot enough images found. Please provide 2 or more images.\n")
        exit(1)
    
    return Images

def Convert_xy(x, y,center,f):
    #global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return xt, yt

def ProjectOnCylinder(im):
    h, w = im.shape[:2]
    center = [w // 2, h // 2]
    f = 700

    ti = np.zeros(im.shape, dtype=np.uint8)

    cord_ti = np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = cord_ti[:, 0]
    ti_y = cord_ti[:, 1]

    ii_x, ii_y = Convert_xy(ti_x, ti_y, center, f)

    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    withinIndex = (ii_tl_x >= 0) * (ii_tl_x <= (w - 2)) * \
                  (ii_tl_y >= 0) * (ii_tl_y <= (h - 2))

    ti_x = ti_x[withinIndex]
    ti_y = ti_y[withinIndex]

    ii_x = ii_x[withinIndex]
    ii_y = ii_y[withinIndex]

    ii_tl_x = ii_tl_x[withinIndex]
    ii_tl_y = ii_tl_y[withinIndex]

    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx) * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx) * (dy)

    ti[ti_y, ti_x, :] = (weight_tl[:, None] * im[ii_tl_y, ii_tl_x, :]) + \
                        (weight_tr[:, None] * im[ii_tl_y, ii_tl_x + 1, :]) + \
                        (weight_bl[:, None] * im[ii_tl_y + 1, ii_tl_x, :]) + \
                        (weight_br[:, None] * im[ii_tl_y + 1, ii_tl_x + 1, :])

    min_x = min(ti_x)

    ti = ti[:, min_x: -min_x, :]

    return ti, ti_x - min_x, ti_y

if __name__ == "__main__":
    Images = ReadImage("Images/Field")
    # Resize the first image
    #Images[0] = cv2.resize(Images[0], (Images[0].shape[1] // 2, Images[0].shape[0] // 2))
   
    baseImage, _, _ = ProjectOnCylinder(Images[5])
    
    cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
    cv2.imshow('Matches', baseImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ci = StitchImages(baseImage, Images[1])

    # cv2.imwrite("Stitched_Panorama.png", ci)
