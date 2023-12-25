# pre-process test images using adaptive equalization
import cv2
import os

# Specify the paths to the input (original) and output (enhanced) folders
input_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/Sample/'
output_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/enhanced_images_3/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to apply adaptive histogram equalization to an image
def apply_adaptive_equalization(image_path, output_path):
    # Read the color image
    img = cv2.imread(image_path)

    # Split the image into color channels
    channels = cv2.split(img)

    # Apply adaptive histogram equalization to each channel independently
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_channels = [clahe.apply(channel) for channel in channels]

    # Merge the enhanced channels back into a color image
    img_enhanced = cv2.merge(enhanced_channels)

    # Save the enhanced image
    cv2.imwrite(output_path, img_enhanced)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Full path to the input and output images
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, f'{filename}')

        # Apply adaptive histogram equalization
        apply_adaptive_equalization(input_image_path, output_image_path)

# Display or further process the enhanced images as needed
