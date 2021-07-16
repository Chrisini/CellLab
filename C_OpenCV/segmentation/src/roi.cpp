/*
 * find_roi.cpp
 *
 *  Created on: 11 Feb 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
using namespace cv;

/**
 * The higher the offset, the smaller the image
 */
Rect roi_getter(){

	Rect roi;
	roi.x = 1;
	roi.y = 1;
	roi.width = 1; // img.size().height/2; // img.size().height - (offset_y);
	roi.height = 1; //img.size().height/2; //img.size().height - (offset_y);
	return roi;
}
