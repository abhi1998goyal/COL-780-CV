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
            ls.append([x_center.item(),y_center.item(),(w/width).item(),(h/height).item(),confidence_score])
            
    return ls


def save_pts_as_csv(image_file, pts):
    image_name = os.path.splitext(image_file)[0]
    output_directory = '/kaggle/working/ditr/test/predictions2'
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join('/kaggle/working/ditr/test/predictions2', f"{image_name}_preds.txt")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        #writer.writerow(['x_center', 'y_center', 'width', 'height', 'score'])
        for pt in pts:
            writer.writerow(pt)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/kaggle/input/ditr-model/ditr_model.pt"
    model = load_model(checkpoint_path)
    model.to(device)
    model.eval()
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    img_folder='/kaggle/input/bc-coco/coco_1k/val2017'
    #image_files = [os.path.join(img_folder, img) for img in os.listdir(img_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = os.listdir(img_folder)

    for image_file in image_files:
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
#         print(image_file)
#         print(pts)
        
        save_pts_as_csv(image_file, pts)


if __name__ == "__main__":
    main()
