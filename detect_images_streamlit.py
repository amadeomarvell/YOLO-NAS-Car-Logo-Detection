import streamlit as st
from super_gradients.training import models
import torch
import cv2
import random
import numpy as np
import os
from glob import glob
from PIL import Image

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # Reduce font size by adjusting fontScale
        font_scale = tl / 4  # Adjust this value to further reduce the font size
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Function to apply adaptive histogram equalization to an image
def apply_adaptive_equalization(img):
    # Split the image into color channels
    channels = cv2.split(img)

    # Apply adaptive histogram equalization to each channel independently
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_channels = [clahe.apply(channel) for channel in channels]

    # Merge the enhanced channels back into a color image
    img_enhanced = cv2.merge(enhanced_channels)

    return img_enhanced

# Function to perform inference
def perform_inference(uploaded_file):
    # Load YOLO-NAS Model
    model = models.get(
        'yolo_nas_m',
        num_classes=28,
        checkpoint_path='/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/ckpt_best.pth'
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
     # Get class names from the model
    class_names = model.predict(np.zeros((1, 1, 3)), conf=confidence)._images_prediction_lst[0].class_names

    img_array = np.array(Image.open(uploaded_file))
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Inference logic (modify this based on your YOLO-NAS script)
    preds = model.predict(img, conf=confidence)._images_prediction_lst[0]
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)

    label_names_confidence = {}
    for box, cnf, cs in zip(bboxes, confs, labels):
        class_name = class_names[int(cs)]
        label_names_confidence[class_name] = cnf

        plot_one_box(box[:4], img, label=f'{class_name} {cnf:.3}', color=[255, 0, 0])

    return img, label_names_confidence

# Streamlit App
# Set the width of the sidebar
st.set_page_config(layout="wide")

st.title("Car Logo Detection with YOLO-NAS")
st.sidebar.title("Settings")
st.sidebar.subheader("Parameters")
confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0)

app_model = st.sidebar.selectbox('Choose the app mode', ('Input Image', 'Select from Folder'))

if app_model == 'Input Image':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Allow users to perform inference on the selected image
    if st.button("Perform Inference"):

        # Perform inference on the selected image
        img, labels = perform_inference(uploaded_file)

        # Display the image with bounding boxes
        st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

        # Display detected labels
        st.write(f"Detected labels: {labels}")

elif app_model == 'Select from Folder':
    # Allow users to choose a local folder
    base_folder_path = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Car_Logo_Dataset_27_Contrasted_COCO_FINAL/test2/'
    
    # Get all files from the selected folder
    files = os.listdir(base_folder_path)

    # Display the list of images in the sidebar
    selected_image = st.sidebar.selectbox("Select an image for inference", files)

    # Full path to the selected image
    selected_image_path = os.path.join(base_folder_path, selected_image)

    # Allow users to perform inference on the selected image
    if st.sidebar.button("Perform Inference"):
        # Perform inference on the selected image
        img, labels = perform_inference(selected_image_path)

        # Display the image with bounding boxes
        st.image(img, channels="BGR", caption="Processed Image", use_column_width=True)

        # Display detected labels
        st.write(f"Detected labels: {labels}")





