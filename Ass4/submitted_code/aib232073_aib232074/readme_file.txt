Instructions for running codes:

1. Faster RCNN model train:

python3 rcnn_detr.py <path_to_train_image_folder> <path_to_val_image_folder>

2. Faster RCNN model test:

python3 train_rcnn.py <path_to_train_image_folder> <path_to_val_image_folder>

3. Deformable DETR model train:

python3 train_detr.py <path_to_train_image_folder> <path_to_val_image_folder>

4. Deformable DETR model test:

python3 test_detr.py <path_to_image_folder> <path_to_trained_model> <path_to_save>