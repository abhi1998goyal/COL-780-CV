import os
import cv2
import numpy as np
import csv
import sys
import pandas as pd

def erode(img, kernel):
    pad_size = kernel.shape[0] // 2
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    eroded_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(pad_size, img.shape[0] + pad_size):
        for j in range(pad_size, img.shape[1] + pad_size):
            eroded_img[i - pad_size, j - pad_size] = np.min(padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1] * kernel)

    return eroded_img

def dilate(img, kernel):
    pad_size = kernel.shape[0] // 2
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
    dilated_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(pad_size, img.shape[0] + pad_size):
        for j in range(pad_size, img.shape[1] + pad_size):
            dilated_img[i - pad_size, j - pad_size] = np.max(padded_img[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1] * kernel)

    return dilated_img

def subtract(img1, img2):
    return np.maximum(img1 - img2, 0)

def bitwise_or(img1, img2):
    return np.maximum(img1, img2)

def threshold_image(image):
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if(image[i][j]>70):
                image[i][j]=0
            else:
                image[i][j]=255
    return image

def get_image_skeleton(img):
     element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
     skel = np.zeros_like(img, dtype=np.uint8)
     done = False
     while not done:
        eroded = erode(img, element)
        temp = dilate(eroded, element)
        temp = subtract(img, temp)
        skel = bitwise_or(skel, temp)
        img = eroded.copy()
        done = (np.max(img) == 0)

     return skel

def hough_transform(image,low_angle,high_angle):
    thetas = np.deg2rad(np.arange(low_angle, high_angle, 1))
    diag_len = int(np.sqrt(image.shape[0]**2 + image.shape[1]**2))
    rhos = np.arange(-diag_len, diag_len, 1)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y, x = np.nonzero(image)
    included_y_val = set()
    min_rho_distance=7

    for i in range(len(x)):
        for j in range(len(thetas)):
            rho = int(x[i] * np.cos(thetas[j]) + y[i] * np.sin(thetas[j]))
            #if not included_y or min(abs(y[i] - included_y_val) for included_y_val in included_y) >= min_rho_distance:
            rho_index = np.argmin(np.abs(rhos - rho))
               # included_y.add(y[i])
            accumulator[rho_index, j] += 1
    
    peaks = []
    sel_param = []
    sorted_indices = np.argsort(accumulator.ravel())[::-1]
    limit=0#25
    
    last_val=accumulator.ravel()[sorted_indices[0]].astype(int)
    for idx in sorted_indices:
        i, j = np.unravel_index(idx, accumulator.shape)
        filtered_y = y[(x * np.cos(thetas[j]) + y * np.sin(thetas[j])).astype(int) == rhos[i]]
        if len(filtered_y) > 0:
           avg_y = np.mean(filtered_y)
        else:
           avg_y = 0  # or any other default value you want to use

        #avg_y = np.mean(y[(x * np.cos(thetas[j]) + y * np.sin(thetas[j])).astype(int) == rhos[i]])
        if not included_y_val or min(abs(avg_y - included_y) for included_y in included_y_val) >= min_rho_distance: 
            peaks.append((i, j))
            limit=limit+1
            included_y_val.add(avg_y)
            print(f'value  {accumulator.ravel()[idx]} \n')
            #if abs(accumulator.ravel()[idx].astype(int) - last_val)> 15 or accumulator.ravel()[idx].astype(int)<35:
            if accumulator.ravel()[idx].astype(int)<38:
               break
            print(f'y  {int(avg_y)} theta {int(90-thetas[j]*57.29)} \n')
            sel_param.append((int(avg_y), thetas[j]))
            last_val = accumulator.ravel()[idx]


    return sel_param


def get_m_c(image,low_angle,high_angle):
    param = hough_transform(image,low_angle,high_angle)
    #param=hough_peaks(accum,25)
    return param


def process_image(image_path):
    #kernel=np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_original = cv2.imread(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = threshold_image(image)
    #image = dilate(image,kernel)
    image = get_image_skeleton(image)
    image = dilate(image,kernel)
    #corners = cv2.cornerHarris(image, blockSize=13, ksize=3, k=0.04)
    param=get_m_c(image,80,100)

    # lines_image=draw_lines(img_original,param)

    # cv2.imshow('Original Image', image)
    # cv2.imshow('Skeleton', lines_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    num_lines = len(param)
    param_sorted = sorted(param, key=lambda x: x[0])
    distances = np.abs(np.diff([rho for rho, _ in param_sorted]))
    distances = distances / image.shape[1]

    mean_distance = round(np.mean(distances),4)
    #std_distance = int(np.std(distances)) 
    var_distance = round(np.sum((distances - mean_distance)**2) / len(distances),4)

    angles = [90-np.rad2deg(theta) for _, theta in param_sorted]
    mean_angle = abs(int(np.mean(angles)))
   # std_angle = int(np.std(angles))
    var_angle = int(np.var(angles))

    return num_lines, mean_distance, var_distance, mean_angle, var_angle


def draw_lines(image, param):
    image_with_lines = np.copy(image)
    for rho, theta in param:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image_with_lines

def task1(image_dir,output_csv):
#def task1():
      with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'num_sutures', 'mean_inter_suture_spacing', 'variance_inter_suture_spacing',
                      'mean_suture_angle', 'variance_suture_angle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for filename in os.listdir(image_dir):
            image_path = os.path.join(image_dir, filename)
        # image_path='data\img9.png'
            num_lines,mean_distance,variance_distance,mean_angle,variance_angle=process_image(image_path)
            writer.writerow({
                                'image_name': filename,
                                'num_sutures': num_lines,
                                'mean_inter_suture_spacing': mean_distance,
                                'variance_inter_suture_spacing': variance_distance,
                                'mean_suture_angle': mean_angle,
                                'variance_suture_angle': variance_angle
                            })

def compare_images(image_path1, image_path2):
    num_lines1, mean_distance1, variance_distance1, mean_angle1, variance_angle1 = process_image(image_path1)
    num_lines2, mean_distance2, variance_distance2, mean_angle2, variance_angle2 = process_image(image_path2)
    distance_winner = 1 if variance_distance1 < variance_distance2 else 2
    angle_winner = 1 if variance_angle1 < variance_angle2 else 2
    return distance_winner, angle_winner

def task2(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['img1_path', 'img2_path', 'output_distance', 'output_angle']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _, row in df.iterrows():
            img1_path = row['img1_path']
            img2_path = row['img2_path']

            distance_winner, angle_winner = compare_images(img1_path, img2_path)

            writer.writerow({
                'img1_path': img1_path,
                'img2_path': img2_path,
                'output_distance': distance_winner,
                'output_angle': angle_winner
            })

if __name__=='__main__':
     if sys.argv[1]=='1':
        image_dir = sys.argv[2]
        output_csv = sys.argv[3]
        task1(image_dir, output_csv)
     if sys.argv[1] == '2':
        input_csv = sys.argv[2]
        output_csv = sys.argv[3]
        task2(input_csv, output_csv)
