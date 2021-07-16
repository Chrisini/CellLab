/*
 * algorithm.cpp
 *
 *  Created on: 7 Jul 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>
#include "math.h"
#include <ctime>
using namespace cv;
using namespace std;

Mat algorithm_cht(Mat ima) {

	Mat src = ima.clone();
	Mat src_gray = ima.clone();




	  /// Convert it to gray
		cvtColor( src, src_gray, CV_BGR2GRAY );

	  /// Reduce the noise so we avoid false circle detection
	  GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

	  src_gray = feature_canny_edge(src_gray);

	  imshow("canny", src_gray);

	  vector<Vec3f> circles;

	  /// Apply the Hough Transform to find the circles
	  HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 15, 30, 1, 0.1, 5 );

	  /// Draw the circles detected
	  for( size_t i = 0; i < circles.size(); i++ )
	  {
	      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	      int radius = cvRound(circles[i][2]);
	      // circle center
	      circle( src, center, 1, Scalar(0,255,0), -1, 8, 0 );
	      // circle outline
	      circle( src, center, radius, Scalar(0,0,255), 1, 8, 0 );
	   }

	  imshow( "Hough Circle Transform Demo", src );


	return src;
}

Mat algorithm_contours(Mat original) {

	return original;
}

Mat algorithm_edge_detection(Mat original) {
	return original;
}

Mat algorithm_graph_based(Mat original) {
	return original;
}


Mat algorithm_watershed(Mat original) {
	return original;
}


