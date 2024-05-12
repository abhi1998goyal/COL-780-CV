from torch.utils.data import Dataset
import torch
import os 
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.images_folder_path = os.path.join(folder_path, 'images')
        self.labels_folder_path = os.path.join(folder_path, 'labels')
        self.transform = transform
        self.filenames = [f for f in os.listdir(self.images_folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def get_all_images_and_boxes(self):
        data = []
        for filename in self.filenames:
            image_path = os.path.join(self.images_folder_path, filename)
            image = cv2.imread(image_path)
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

            label_file_path = os.path.join(self.labels_folder_path, filename.replace('.png', '.txt').replace('.jpg', '.txt'))
            boxes = []
            if os.path.isfile(label_file_path):
                with open(label_file_path, 'r') as file:
                    for line in file:
                        label, x_center, y_center, width, height = map(float, line.strip().split(' '))
                        x_min = x_center - width / 2
                        y_min = y_center - height / 2
                        x_max = x_min + width
                        y_max = y_min + height
                        boxes.append([label, x_min, y_min, x_max, y_max])

            if boxes:
                labels = torch.tensor([b[0] for b in boxes], dtype=torch.int64)
                boxes = torch.tensor(boxes, dtype=torch.float32)
                boxes = boxes[:, 1:]
                data.append((image.to(device), {'boxes': boxes.to(device), 'labels': labels.to(device)}))

        return data