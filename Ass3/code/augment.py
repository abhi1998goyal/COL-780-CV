import os
import cv2
import numpy as np
import random
import cropped_rectangle

# Define the input and output folders
input_folder = 'Final_dataset/open_hands'
output_folder = 'Final_dataset/open_hands/color_aug'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Define the augmentation parameters
# You can adjust these parameters according to your requirements
rotation_angles = [-20, 20]  # Rotation angles in degrees
flip_modes = [1, -1]  # 0 - no flip, 1 - horizontal flip, -1 - vertical flip
translation_ranges = [(-50, 50)]  # Translation ranges in pixels (x, y)
crop_sizes = [(0.9, 0.9),(0.85, 0.85),(0.8,0.8)]  # Crop sizes as a percentage of original image size (width, height)
scale_factors = [0.5,2, 4.2]  # Scale factors for zooming in/out
noise_levels = [20, 40]  # Noise levels (standard deviation)
color_jitters = [(0.8, 1.2 , 0.5), (1.2, 0.5,0.8), (0.5, 0.8,1.2)]  # Color jittering factors for brightness, contrast, and saturation

# Apply augmentation and save the augmented images
for file in files:
    # Read the image
    try:
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        image,(yg,xg)=cropped_rectangle.get_green_window(image)

    # Apply rotation
    # for angle in rotation_angles:
    # # Compute the rotation matrix
    #     rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)
    #     # Apply the rotation transformation
    #     rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    #     # Write the rotated image to file
    #     output_file = os.path.splitext(file)[0] + f'_rotated_{angle}.jpg'
    #     cv2.imwrite(os.path.join(output_folder, output_file), rotated_image)



    #Apply flipping
        for flip_mode in flip_modes:
            flipped_image = cv2.flip(image, flip_mode)
            flip_name = 'horizontal' if flip_mode == 1 else 'vertical' if flip_mode == -1 else 'no'
            output_file = os.path.splitext(file)[0] + '_flipped_' + flip_name + '.jpg'
            cv2.imwrite(os.path.join(output_folder, output_file), flipped_image)

    # Apply translation
    # for translation_range in translation_ranges:
    #     x_translation = random.randint(translation_range[0], translation_range[1])
    #     y_translation = random.randint(translation_range[0], translation_range[1])
    #     translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    #     translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    #     output_file = os.path.splitext(file)[0] + f'_translated_{x_translation}_{y_translation}.jpg'
    #     cv2.imwrite(os.path.join(output_folder, output_file), translated_image)

    #Apply random cropping
    # for crop_size in crop_sizes:
    #     crop_width = int(image.shape[1] * crop_size[0])
    #     crop_height = int(image.shape[0] * crop_size[1])
    #     x_start = random.randint(0, image.shape[1] - crop_width)
    #     y_start = random.randint(0, image.shape[0] - crop_height)
    #     cropped_image = image[y_start:y_start + crop_height, x_start:x_start + crop_width]
    #     output_file = os.path.splitext(file)[0] + f'_cropped_{crop_width}x{crop_height}.jpg'
    #     cv2.imwrite(os.path.join(output_folder, output_file), cropped_image)

    # Apply scaling
    # for scale_factor in scale_factors:
    #     scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    #     output_file = os.path.splitext(file)[0] + f'_scaled_{scale_factor}.jpg'
    #     cv2.imwrite(os.path.join(output_folder, output_file), scaled_image)

    # Apply noise
    # for noise_level in noise_levels:
    #     noisy_image = np.uint8(np.clip(image + np.random.normal(0, noise_level, image.shape), 0, 255))
    #     output_file = os.path.splitext(file)[0] + f'_noisy_{noise_level}.jpg'
    #     cv2.imwrite(os.path.join(output_folder, output_file), noisy_image)

    #Apply color jitter
        # for jitter_factors in color_jitters:
        #     jittered_image = (image.astype(np.float32) * np.array(jitter_factors)).clip(0, 255).astype(np.uint8)
        #     output_file = os.path.splitext(file)[0] + f'_jittered_{jitter_factors[0]}_{jitter_factors[1]}_{jitter_factors[2]}.jpg'
        #     cv2.imwrite(os.path.join(output_folder, output_file), jittered_image)
    except:
        print('skipped')


print("Augmentation completed.")
