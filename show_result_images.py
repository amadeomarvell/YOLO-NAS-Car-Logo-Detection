import os
from PIL import Image
import matplotlib.pyplot as plt

results_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/runs/detect/yolo_logo_det/'
warped_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLOv7-Car-Logo-Detection/data/warped_plates/'

# Get a list of image filenames
image_filenames = [filename for filename in os.listdir(results_path) if filename.endswith('.jpg')]

# Iterate through the pairs
for filename in image_filenames:
    # Get the paths for the source and warped images
    source_path = os.path.join(results_path, filename)
    warped_path = os.path.join(warped_folder, filename)

    # Open the source and warped images
    source_image = Image.open(source_path)
    warped_image = Image.open(warped_path)

    # Display the source and warped images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Display source image on the left side
    axs[0].imshow(source_image)
    axs[0].axis('off')

    # Display warped image on the right side
    axs[1].imshow(warped_image)
    axs[1].axis('off')

    plt.show()

    # Ask the user to press Enter to proceed to the next pair
    input('Press Enter to show the next pair...')

