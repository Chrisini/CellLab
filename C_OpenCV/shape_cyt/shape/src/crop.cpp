// code from here: https://stackoverflow.com/questions/14365411/opencv-crop-image/14365605
#include "opencv2/highgui/highgui.hpp" // not sure about this
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;
Mat image=imread("image.png",1);

int startX=200,startY=200,width=100,height=100

Mat ROI(image, Rect(startX,startY,width,height));

Mat croppedImage;

// Copy the data into new matrix
ROI.copyTo(croppedImage);

imwrite("newImage.png",croppedImage);
