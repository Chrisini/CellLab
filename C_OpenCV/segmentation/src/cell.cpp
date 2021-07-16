/*
 * crop_images.cpp
 *
 *  Created on: 11 Feb 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
using namespace cv;




/**
 *  Crop the original image to the defined ROI
 */
Mat cell_crop(Mat input, Rect roi){

    Mat output = input(roi);
    return output;

}
