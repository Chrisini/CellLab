/*
 * morphology.cpp
 *
 *  Created on: 25 Feb 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
using namespace cv;

Mat morph_erode(Mat input) {
	Mat output, element;
	int morph_size = 1;
	element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	erode( input, output, element);
	return output;
}

Mat morph_dilate(Mat input) {
	Mat output, element;
	int morph_size = 3;
	element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
	dilate(input, output, element);
	return output;
}

Mat morph_open(Mat input) {
	Mat output, element;
	int morph_size = 6;
	element = getStructuringElement( MORPH_ELLIPSE, Size(2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ));
	morphologyEx(input, output, MORPH_OPEN, element );
	return output;
}

Mat morph_close(Mat input) {
	Mat output, element;
	int morph_size = 1;
	element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ));
	morphologyEx(input, output, MORPH_CLOSE, element );
	return output;
}

Mat morph_fill(Mat input){
	Mat output = input.clone();
    floodFill(output, Point(2,2), Scalar(255));
    return output;
}




