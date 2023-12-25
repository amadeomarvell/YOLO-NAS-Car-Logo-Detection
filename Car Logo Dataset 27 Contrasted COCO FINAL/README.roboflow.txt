
Test 1 - v6 2023-12-25 1:25pm
==============================

This dataset was exported via roboflow.com on December 25, 2023 at 6:30 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 6605 images.
Car-logos are annotated in COCO format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 480x480 (Stretch)
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 2 versions of each source image:
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random brigthness adjustment of between -2 and +2 percent
* Random exposure adjustment of between -2 and +2 percent
* Salt and pepper noise was applied to 1 percent of pixels


