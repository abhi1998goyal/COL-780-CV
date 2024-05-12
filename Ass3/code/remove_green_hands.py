import os
import cv2
import cropped_rectangle

input_folder = 'Final_dataset/open2/train'
output_folder = 'Final_dataset/precessed_open_hands'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = os.listdir(input_folder)
#sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
#sorted_files = sorted(files) 
# Apply augmentation and save the augmented images
for file in files:
    # if(file.startswith('670')):
    #     break
    try:
    # Read the image
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        image,(yg,xg)=cropped_rectangle.get_green_window(image)

        cv2.imwrite(os.path.join(output_folder, file), image)

    except Exception as e:
        print(f"Error processing {file}: {e}")