import cv2
import cropped_rectangle
import numpy as np
from skimage.feature import hog

def normalize_hog_cell(hog_cell):
    hog_cell_norm = np.linalg.norm(hog_cell)
    normalized_hog_cell = hog_cell / (hog_cell_norm + 1e-5)  

    return normalized_hog_cell


def visualize_hog(hog_features, cell_size=(8, 8), orientations=9):
    num_cells_x = hog_features.shape[0]
    num_cells_y = hog_features.shape[1]

    pixels_per_cell_x = cell_size[0]
    pixels_per_cell_y = cell_size[1]

    image_height = num_cells_y * pixels_per_cell_y
    image_width = num_cells_x * pixels_per_cell_x

    hog_image = np.zeros((image_height, image_width))

    cell_center_x = pixels_per_cell_x // 2
    cell_center_y = pixels_per_cell_y // 2

    for y in range(num_cells_y):
        for x in range(num_cells_x):
            hog_cell =hog_features[x, y, :]
            center_x = x * pixels_per_cell_x + cell_center_x
            center_y = y * pixels_per_cell_y + cell_center_y
            
            total_magnitude = np.sum(hog_cell)
            
            if total_magnitude > 0:
                hog_cell /= total_magnitude

            for i in range(orientations):
               # i= np.argmax(hog_cell)
                angle = i * (180 / orientations)
                
                dx = np.cos(np.radians(angle)) * hog_cell[i]
                dy = np.sin(np.radians(angle)) * hog_cell[i]
                x1 = int(center_x - dx * cell_size[0] / 2)
                y1 = int(center_y - dy * cell_size[1] / 2)
                x2 = int(center_x + dx * cell_size[0] / 2)
                y2 = int(center_y + dy * cell_size[1] / 2)
                
                cv2.line(hog_image, (x1, y1), (x2, y2), color=int(255**hog_cell[i]), thickness=1)
    

    return hog_image

def compute_gradients(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)
    angle[angle < 0] += 180 
    
    return magnitude, angle

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 128))
    return resized


def calc_hog(magnitude,angle,cell_size,num_bins=9):
    cell_wid=magnitude.shape[1] // cell_size
    cell_height=magnitude.shape[0] // cell_size
    cell_hist = np.zeros((cell_wid, cell_height, num_bins))
    
    cell_vec = np.zeros((cell_wid-1, cell_height-1, 36))
    
    for i in range(0,cell_wid):
        for j in range(0,cell_height):
            angle_cell=angle[j*cell_size:(j+1)*cell_size,i*cell_size:(i+1)*cell_size]
            magnitude_cell=magnitude[j*cell_size:(j+1)*cell_size,i*cell_size:(i+1)*cell_size]
            cell_hist[i,j],_= np.histogram(angle_cell, bins=num_bins, range=(0, 180), weights=magnitude_cell)
            
        
    for i in range(cell_wid-1):
        for j in range(cell_height-1):
            cell_vec[i, j] = normalize_hog_cell(np.concatenate((cell_hist[i, j], cell_hist[i+1, j], cell_hist[i, j+1], cell_hist[i+1, j+1])))
    
    return cell_hist  , cell_vec.flatten(order='C') 
                
            
    
#image = cv2.imread('Final_dataset/open1/train/image_1.jpg')


#test_image=preprocess_image(test_image)

def calc_img_hog1(image):
    block_features=[]
    try:
        cropped_img,_=cropped_rectangle.get_green_window(image)
        gray_img=preprocess_image(cropped_img)
        grad_mag,grad_angle=compute_gradients(gray_img)


        hog_features,block_features = calc_hog(grad_mag, grad_angle, cell_size=8)
    except:
        print('could not get features')

    return block_features


def calc_img_hog2(image):

    #cropped_img=cropped_rectangle.get_green_window(image)
    gray_img=preprocess_image(image)
    grad_mag,grad_angle=compute_gradients(gray_img)

    hog_features,block_features = calc_hog(grad_mag, grad_angle, cell_size=8)

    return block_features

#hog_image = visualize_hog(hog_features)
#fd, hog_image = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True)

# Display the HOG image
#hog_image = cv2.resize(hog_image, (64 * 4, 128 * 4))
#cv2.imshow('HOG Image', hog_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imshow('Detected Hands', gray_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()