/*
 * edge.cpp
 *
 *  Created on: 25 Feb 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>
#include "math.h"
#include <ctime>
using namespace cv;
using namespace std;


Mat feature_canny_edge(Mat input) {

	Mat grey, output, detected_edges;
	int lowThreshold = 50;
	int ratio = 30;
	int kernel_size = 5;

	// grey = colour_grey(input);

	/// Reduce noise with a kernel 3x3
	// blur(grey, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(input, detected_edges, lowThreshold, lowThreshold * ratio,
			kernel_size);

	/// Using Canny's output as a mask, we display our result
	output = Scalar::all(0);

	input.copyTo(output, detected_edges);

	return output;

}

Mat feature_ridge(Mat input) {
	Mat output;
	/*cv::ximgproc::RidgeDetectionFilter::create(CV_32FC1, 1, 1, 3, CV_8UC1, 1, 0, BORDER_DEFAULT);
	 cv::ximgproc::RidgeDetectionFilter::getRidgeFilteredImage(input, output);
	 //cv::ximgproc::RidgeDetectionFilter::getRidgeFilteredImage*/
	return output;

}

Mat feature_contour(Mat input, int threshnumber, int particle_size) {

	RNG rng(12345);
	Mat output, src_gray;
	Mat canny_output;
	vector < vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Convert image to gray and blur it
	if (input.channels() != 1) cvtColor(input, input, CV_BGR2GRAY);
	blur(input, src_gray, Size(3, 3));

	// 150, 200 work good // or 160, 200
	// 20, 200 if got rid of gradient
	threshold(src_gray, canny_output, threshnumber, 200, THRESH_BINARY);

	imshow("Thresh", canny_output);

	findContours(canny_output, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));

	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	for (int i = 0; i < contours.size(); i++) {
		if(contours.at(i).size() > particle_size){
			Scalar colour[4];
			colour[0] = Scalar(255, 200, 50); // blue
			colour[1] = Scalar(50, 255, 100); // green
			colour[2] = Scalar(50, 100, 255); // red
			colour[3] = Scalar(255, 50, 230); // pink
			drawContours(drawing, contours, i, colour[rng.uniform(0,4)], 1, 8, hierarchy, 0, Point());
		}
		}
	imshow("contours", drawing);



	return drawing;
}


/*
 * Code for finding contours
 * first thresholding then:
 */

/*
	findContours(fibres, contours, hierarchy, CV_RETR_TREE,
			CV_CHAIN_APPROX_TC89_KCOS, Point(0, 0));

	// empty Mat, that will be used to draw the contours on
	mask = Mat::zeros(fibres.size(), CV_8UC3);

	for (int i = 0; i < contours.size(); i++) {
		if (contours.at(i).size() > particleval) {
			Scalar colour[4];
			colour[0] = Scalar(255, 200, 50); // blue
			colour[1] = Scalar(50, 255, 100); // green
			colour[2] = Scalar(50, 100, 255); // red
			colour[3] = Scalar(255, 50, 230); // pink
			drawContours(mask, contours, i, colour[rng.uniform(0, 4)], 1, 8,
					hierarchy, 0, Point()); // use drawing instead of original
		}
	}

	imshow("Threshold (fibre)", fibres);
	imshow("Mask (fibre)", mask); // use drawing for black background
*/
