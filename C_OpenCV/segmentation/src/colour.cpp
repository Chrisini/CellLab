/*
 * colour.cpp
 *
 *  Created on: 25 Feb 2020
 *      Author: Prinzessin
 */
#include "../inc/segmentation.hpp"
using namespace cv;

Mat colour_grey(Mat input){

	Mat output;
	cvtColor(input, output, COLOR_BGR2GRAY);
	return output;

}


