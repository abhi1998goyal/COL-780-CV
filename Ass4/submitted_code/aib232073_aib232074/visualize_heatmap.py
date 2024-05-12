#pip install --upgrade --force-reinstall pytorch-grad-cam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch 
import cv2
import os 
import csv
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import sys
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    #model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    model = fasterrcnn_resnet50_fpn()
    # checkpoint = torch.load(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def create_predicted_boxes_images(predictions,image_height,image_width):
    boxes=predictions['boxes']
    scores=predictions['scores']
    pts=[]

    for i in range(len(predictions['boxes'])):
        x_min, y_min, x_max, y_max =boxes[i]
        score=scores[i]
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        pts.append([x_center.item(), y_center.item(), (width).item(), (height).item(),score.item()])

    return pts

def save_pts_as_csv(image_file, pts, output_directory):
    image_name = os.path.splitext(image_file)[0]
    output_directory = output_directory
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f"{image_name}_preds.txt")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
       # writer.writerow(['x_center', 'y_center', 'width', 'height', 'score'])
        for pt in pts:
            writer.writerow(pt)

def main():
    # model_path = '/kaggle/input/finetunepretrained-rcnn/model.pt'
    model_path = sys.argv[2]
    # model_path = "model_rcnn.pt"
    # images_path = '/kaggle/input/froc-test/test/images'
    images_path = sys.argv[3]
    # images_path = "image"
    output_directory = sys.argv[4]
    # output_directory = "image"
    checkpoint_path = model_path
    model = load_model(checkpoint_path)
    model.to(device)
    model.eval()
    target_layers = [model.backbone]
    model = model.to(torch.device('cpu'))

    cam = EigenCAM(model,
              target_layers, 
            #   use_cuda=torch.nn.cuda.is_available(), 
            #   use_cuda=False,
              reshape_transform=fasterrcnn_reshape_transform)

    image_files = os.listdir(images_path)
    for image_file in image_files: 
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)
       # print(image.shape)
        image_height, image_width, _ = image.shape
        
        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)[0]
            # print(predictions)
            grayscale_cam = cam(image_tensor, targets=predictions)
            grayscale_cam = grayscale_cam[0, :]

            plt.imshow(grayscale_cam)

            plt.show()
            pts=create_predicted_boxes_images(predictions,image_height,image_width)
            # print(image_file)
            # print(pts)
     



if __name__ == "__main__":
    main()
