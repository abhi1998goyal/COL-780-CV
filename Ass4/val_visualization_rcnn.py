import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch 
import cv2
import os 
import torchvision.ops as ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path):
    #model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    model = fasterrcnn_resnet50_fpn()
    #checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))# when only cpu is present
    checkpoint = torch.load(checkpoint_path) #!uncomment above if device is gpu 
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
    
def apply_nms(predictions, threshold=0):
    boxes = predictions['boxes']
    scores = predictions['scores']
    keep = ops.nms(boxes, scores, threshold)
    predictions['boxes'] = boxes[keep]
    predictions['scores'] = scores[keep]
    print( predictions['scores'])
    return predictions

def create_predicted_boxes_images(image, predictions):
    for box,score in zip(predictions['boxes'],predictions['scores']):
        if(score>0):
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

    return image

def  create_ground_boxes_images(image, ground_boxes):
    for box in ground_boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255,0 , 0), 2)

    return image

def test(model_path,images_path):
    checkpoint_path = model_path
    model = load_model(checkpoint_path)
    model.to(device)
    model.eval()

    image_files = os.listdir(images_path)
    for image_file in image_files: 
        #print(image_file)
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)
        label_file_path = os.path.join('/kaggle/input/cv-a4-data/yolo_1k/val/labels', image_file.replace('.png', '.txt').replace('.jpg', '.txt'))
        ground_boxes=[]
        
        if os.path.isfile(label_file_path):
                with open(label_file_path, 'r') as file:
                    for line in file:
                        label, x_center, y_center, width, height = map(float, line.strip().split(' '))
                        x_min = int((x_center - width / 2) * image.shape[1])
                        y_min = int((y_center - height / 2) * image.shape[0])
                        x_max = int((x_center + width / 2) * image.shape[1])
                        y_max = int((y_center + height / 2) * image.shape[0])
                        ground_boxes.append([x_min, y_min, x_max, y_max])
        #print(ground_boxes)

        transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(image_tensor)[0]
            predictions = apply_nms(predictions)
            img1 = create_predicted_boxes_images(image, predictions)
            img2 = create_ground_boxes_images(img1,ground_boxes)
            #print(img2)
            #print('HI')

            output_folder = '/kaggle/working/val'
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, img2)
            #cv2.imwrite(f'/kaggle/working/val/{image_file}', img2)
            plt.figure(figsize=(8, 8))
            plt.imshow(img2)
            plt.show()
            


if __name__ == "__main__":
    test('/kaggle/input/finetunepretrained-rcnn/model.pt','/kaggle/input/cv-a4-data/yolo_1k/val/images')
