
import joblib
import train_svm
import hog
import cv2
import cropped_rectangle

clf = joblib.load('svm_model_5.pkl')
min_size = (64, 128)
scale_factor = 2

def calculate_overlap(rect1, rect2):
    x1 = max(rect1[0][0], rect2[0][0])
    y1 = max(rect1[0][1], rect2[0][1])
    x2 = min(rect1[1][0], rect2[1][0])
    y2 = min(rect1[1][1], rect2[1][1])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_rect1 = (rect1[1][0] - rect1[0][0] + 1) * (rect1[1][1] - rect1[0][1] + 1)
    area_rect2 = (rect2[1][0] - rect2[0][0] + 1) * (rect2[1][1] - rect2[0][1] + 1)
    overlap = intersection / float(area_rect1 + area_rect2 - intersection)
    return overlap

def non_max_suppression(rectangles, overlap_threshold):
    sorted_rectangles = sorted(rectangles, key=lambda x: x[1], reverse=True)
    final_rectangles = []
    #final_rectangles.append(sorted_rectangles[0])
    for rect in sorted_rectangles:
        overlap = False
        for final_rect in final_rectangles:
            if calculate_overlap(rect[0], final_rect) > overlap_threshold:
                overlap = True
                break
        if not overlap:
            if(rect[1]>0.9):
               final_rectangles.append(rect[0])
    return final_rectangles


def image_pyramid(image, min_size, scale_factor):
    x=1
    pyramid = []
    pyramid.append(image)
    while True:
        resized_image = cv2.resize(image, (int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor)))
        if resized_image.shape[0] < min_size[1] or resized_image.shape[1] < min_size[0]:
            break
        pyramid.append(resized_image)
        image = resized_image
        x+=1
    print(f'Total pyramid height {x} \n')
    return pyramid


coord_img=[]
def detect_hand(level,img):
   n=0
   #img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
   if img.shape[0]>min_size[1] and img.shape[1]>min_size[0]:
      for i in range(0,img.shape[1]-min_size[0],10):
         for j in range(0,img.shape[0]-min_size[1],10):
            print(f'{n} \n')
            img_check=img[j:j+min_size[1],i:i+min_size[0]]
            hog_desc = hog.calc_img_hog2(img_check)
            
            if(train_svm.test_hand(clf,hog_desc)[0]==1):
               original_i = int(i * (scale_factor ** level))
               original_j = int(j * (scale_factor ** level))
               min_1_dis =  int(min_size[0] * (scale_factor ** level))
               min_2_dis =  int(min_size[1] * (scale_factor ** level))
               coord_img.append((((original_i, original_j), (original_i + min_1_dis, original_j + min_2_dis)),train_svm.test_hand(clf,hog_desc)[1]))
               #coord_img.append(((i, j), (i + min_size[0], j + min_size[1])))
               #cv2.rectangle(img,(original_i, original_j), (original_i + min_1_dis, original_j + min_2_dis), (0, 0, 255), 1)
               cv2.rectangle(img,(i, j), (i + min_size[0], j + min_size[1]), (0, 0, 255), 1)
            n+=1

      cv2.imshow('Image', img)
      cv2.waitKey(0)
      cv2.destroyAllWindows()


image = cv2.imread('result_image.jpg')
#image = cv2.resize(image,(400,500))

pyramid = image_pyramid(image, min_size, scale_factor)



#cropped_img,(yg,xg)=cropped_rectangle.get_green_window(image)
for i, im in enumerate(pyramid):
    #cv2.imshow(f'Level {i}', im)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    detect_hand(i,im)

overlap_threshold = 0.1
final_rectangles = non_max_suppression(coord_img, overlap_threshold)

for st,ed in final_rectangles:
    cv2.rectangle(image, st, ed, (0, 0, 255), 1)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 810, 1080)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()