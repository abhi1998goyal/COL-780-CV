import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
import os
import csv
import sys
import numpy as np
import cv2          
import time
from torchvision import transforms
import torch
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from transformers import DeformableDetrForObjectDetection, AutoImageProcessor
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

from torch.utils.data import DataLoader
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

def load_model(checkpoint_path):
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def apply_nms(predictions, threshold=0):
    if(len(predictions)>0):
            boxes = torch.tensor([pred[:4] for pred in predictions], dtype=torch.float32)
            scores = torch.tensor([pred[4] for pred in predictions], dtype=torch.float32)

            keep = ops.nms(boxes, scores, threshold)

            predictions = [predictions[i] for i in keep]
    return predictions

def create_predictions(image, logits, pred_boxes, threshold=0.8):
    width, height = image.size
    logits = logits.cpu()
    pred_boxes = pred_boxes.cpu()
    ls=[]
    for box, score  in zip(pred_boxes, logits):
        if score.item() > threshold:
            x1, y1, w, h = box
            x1=x1*width
            w=w*width
            y1=y1*height
            h=h*height
            x1=x1-w/2
            y1=y1-h/2
            x2, y2 = x1 + w, y1 + h
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            confidence_score = score.item()
            ls.append([x1.item(),y1.item(),x2.item(),y2.item(),confidence_score])
            
    return ls

def create_predicted_boxes_images(image, predictions):
    for pred in predictions:
        if(pred[-1]>0):  #confidence
            x_min, y_min, x_max, y_max,_= pred
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    return image

def  create_ground_boxes_images(image, ground_boxes):
    for box in ground_boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0 , 0), 2)

    return image

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/kaggle/input/ditr-model/ditr_model.pt"
    model = load_model(checkpoint_path)
    model.to(device)
    model.eval()
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    img_folder='/kaggle/input/cv-a4-data/yolo_1k/val/images'
    #image_files = [os.path.join(img_folder, img) for img in os.listdir(img_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = os.listdir(img_folder)

    for image_file in image_files:
        
        label_file_path = os.path.join('/kaggle/input/cv-a4-data/yolo_1k/val/labels', image_file.replace('.png', '.txt').replace('.jpg', '.txt'))
        ground_boxes=[]
        image_c = cv2.imread(os.path.join(img_folder, image_file))
        
        if os.path.isfile(label_file_path):
                with open(label_file_path, 'r') as file:
                    for line in file:
                        label, x_center, y_center, width, height = map(float, line.strip().split(' '))
                        x_min = int((x_center - width / 2) * image_c.shape[1])
                        y_min = int((y_center - height / 2) * image_c.shape[0])
                        x_max = int((x_center + width / 2) * image_c.shape[1])
                        y_max = int((y_center + height / 2) * image_c.shape[0])
                        ground_boxes.append([x_min, y_min, x_max, y_max])
                        
        image = Image.open(os.path.join(img_folder, image_file))
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs['pixel_values'])
        logits = outputs['logits']
        
        pred_boxes = outputs['pred_boxes']
        logits = torch.nn.functional.softmax(logits, dim=2)
      
        tensor_np = logits.cpu().numpy()

        max_indices = tensor_np.argmax(axis=2)
    
        keep_indices = (max_indices == 0)

        keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.bool)

        filtered_logits = logits[keep_indices_tensor]
        filtered_pred_boxes = pred_boxes[keep_indices_tensor.unsqueeze(2).expand_as(pred_boxes)].view(-1, 4)

        filtered_logits = filtered_logits[:, 0].unsqueeze(1)

        pts=create_predictions(image, filtered_logits, filtered_pred_boxes)
        #pint(pts)
        
        pts = apply_nms(pts)
        
        
        img1=create_predicted_boxes_images(image_c, pts)
        img2= create_ground_boxes_images(img1,ground_boxes)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img2)
        plt.show()



if __name__ == "__main__":
    main()
