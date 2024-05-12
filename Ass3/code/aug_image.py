import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import numpy as np
import random


# Load the original hand image
image = cv2.imread('Final_dataset/open1/train/image_1.jpg')

# Define the size of the random patch (e.g., 75% of the image)
patch_height = int(0.7 * image.shape[0])
patch_width = int(0.7 * image.shape[1])

# Generate random coordinates for the top-left corner of the patch
start_y = random.randint(0, image.shape[0] - patch_height)
start_x = random.randint(0, image.shape[1] - patch_width)

# Extract the random patch from the original image
random_patch = image[start_y:start_y + patch_height, start_x:start_x + patch_width]

# Visualize the random patch
cv2.imshow('Random Patch', random_patch)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the random patch
#cv2.imwrite('random_patch.jpg', random_patch)
