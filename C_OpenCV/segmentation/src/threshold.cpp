/*
 * threshold.cpp
 *
 *  Created on: 25 Feb 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
using namespace cv;

Mat threshold_adaptive(Mat input){
	Mat output;
	adaptiveThreshold(input, output, 240, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 23, 2);
	return output;
}



