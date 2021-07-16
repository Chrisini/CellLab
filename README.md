# CellLab
Bachelor's project





## Java / OpenIMAJ / Maven
* PreProcessing.java: image normalisation; illuminance correction/background subtraction
* this is the code I used for my thesis

## C / OpenCV
* a real big mess

## MATLAB / Image Processing Toolbox
* the circle hough transform works quite well

## Python / Misc
* preprocessing.py: numpy minmax normalisation, pillow autocontrast normalisation, pillow equalisation, show and save histogram
* SDU19 was a summer school, focusing on deep learning

## Python / TensorFlow / Mask R-CNN matterport implementation
* weak labels, only trained with dead/inhibited cells
* there are issues with the order of cells / mAP is calculated wrong, I think
* works better when zoomed in
* having issues with the cropping, it did some weird streching instead
* this is the code I used for my thesis

### Binary Masks and MS COCO - data format
* Used binary masks from computer vision algorithms and COCO for the manually annotated images
