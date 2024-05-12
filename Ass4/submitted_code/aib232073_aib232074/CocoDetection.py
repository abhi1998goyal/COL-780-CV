import torchvision

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = '/kaggle/input/cv-a4-data/coco_1k/coco_1k/annotations/instances_train2017.json' if train else '/kaggle/input/cv-a4-data/coco_1k/coco_1k/annotations/instances_val2017.json'
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx] 
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() 
        target = encoding["labels"][0] 

        return pixel_values, target