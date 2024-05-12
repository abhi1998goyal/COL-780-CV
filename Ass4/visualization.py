import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import csv
import sys
import cv2

def visualize_image_with_boxes(image_path, label_path):
    image = cv2.imread(image_path)
    boxes = []
    with open(label_path, 'r') as file:
        for line in file:
            label, x_center, y_center, width, height = map(float, line.strip().split(' '))
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])
            boxes.append([x_min, y_min, x_max, y_max])
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    
file_label = 'Calc-Training_P_00034_RIGHT_CC.txt' 
image_path = os.path.join('/kaggle/input/cv-a4-data/yolo_1k/val/images', file_label.replace('.txt', '.png'))
label_path = os.path.join('/kaggle/input/cv-a4-data/yolo_1k/val/labels', file_label)

#visualize_image_with_boxes(image_path, label_path)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch 
import cv2
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def visualize_predicted_boxes(image, predictions):
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box in predictions['boxes']:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

def create_predicted_boxes_images(image, predictions):
    for box in predictions['boxes']:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    return image

def test(model_path,images_path):
    checkpoint_path = "/kaggle/working/rcnn_model.pt"
    model = load_model(checkpoint_path)
    model.to(device)
    model.eval()

    image_files = os.listdir(images_path)
    for image_file in image_files: 
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)

        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)[0]
            img = create_predicted_boxes_images(image, predictions)
            cv2.imwrite('/kaggle/working/yolo_1k/val/images/result',img)


if __name__ == "__main__":
    test('/kaggle/working/rcnn_model.pt','/kaggle/working/yolo_1k/val/images')
