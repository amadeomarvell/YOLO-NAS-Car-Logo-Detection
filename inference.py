from super_gradients.training import models
import torch
import cv2
import random
import numpy as np
import time
import argparse
import os
from glob import glob

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num", type=int, required=True,
                help="number of classes the model trained on")
ap.add_argument("-m", "--model", type=str, default='yolo_nas_s',
                choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
                help="Model type (eg: yolo_nas_s)")
ap.add_argument("-w", "--weight", type=str, required=True,
                help="path to trained model weight")
# ap.add_argument("-s", "--source", type=str, required=True,
#                 help="video path/cam-id/RTSP")
ap.add_argument("-c", "--conf", type=float, default=0.25,
                help="model prediction confidence (0<conf<1)")
ap.add_argument("--save", action='store_true',
                help="Save video")
ap.add_argument("--hide", action='store_false',
                help="to hide inference window")
args = vars(ap.parse_args())


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


def get_bbox(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = model.predict(img_rgb, conf=args['conf'])._images_prediction_lst[0]
    # class_names = preds.class_names
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    for box, cnf, cs in zip(bboxes, confs, labels):
        plot_one_box(box[:4], img, label=f'{class_names[int(cs)]} {cnf:.3}', color=colors[cs])
    return labels


# Load YOLO-NAS Model
model = models.get(
    args['model'],
    num_classes=args['num'], 
    checkpoint_path=args["weight"]
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
class_names = model.predict(np.zeros((1,1,3)), conf=args['conf'])._images_prediction_lst[0].class_names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# Global Timer
global_timer = time.time()

# Inference Image
# Get the list of JPG files in the specified folder
source_folder = '/Users/amade/OneDrive/Desktop/SKRIPSI/YOLO-NAS-Car-Logo-Detection/Car_Logo_Dataset_27_Contrasted_COCO_FINAL/test/'
jpg_files = glob(os.path.join(source_folder, '*.jpg'))

# Loop through each JPG file and perform inference
for jpg_file in jpg_files:
    # Inference on each image
    img = cv2.imread(jpg_file)

    # Debug: Print image shape
    print(f"Image shape: {img.shape}")

    labels = get_bbox(img)

    # Debug: Print detected labels
    print(f"Detected labels: {labels}")

    # Save Image
    if args['save'] or args['hide'] is False:
        os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
        filename = os.path.split(jpg_file)[1]
        path_save = os.path.join('runs', 'detect', filename)
        cv2.imwrite(path_save, img)
        print(f"\033[1m[INFO] Saved Image: {path_save}\033[0m")

        # Debug: Print a message when an image is saved
        print(f"\033[1m[INFO] Saved Image: {path_save}\033[0m")


    # Timer
print(f'[INFO] Completed in \033[1m{(time.time()-global_timer)/60} Minutes\033[0m')