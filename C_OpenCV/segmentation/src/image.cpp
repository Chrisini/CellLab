/*
 * read_images.cpp
 *
 *  Created on: 11 Feb 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
using namespace std;
using namespace cv;

Mat image_read(){
	Mat output;
	output = imread( "data/usb/alive_multi.jpg", 1 );
	if ( !output.data )
	{
		printf("No image data \n");
	}
	return output;
}

void image_show(Mat input){
	namedWindow("Display Cell", WINDOW_AUTOSIZE );
	imshow("Display Cell", input);
	waitKey(0);
}

void image_show_hist(Mat input){

	  /// Separate the image in 3 places ( B, G and R )
	  vector<Mat> bgr_planes;
	  split( input, bgr_planes );

	  /// Establish the number of bins
	  int histSize = 256;

	  /// Set the ranges ( for B,G,R) )
	  float range[] = { 0, 256 } ;
	  const float* histRange = { range };

	  bool uniform = true; bool accumulate = false;

	  Mat b_hist, g_hist, r_hist;

	  /// Compute the histograms:
	  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
	  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	  // Draw the histograms for B, G and R
	  int hist_w = 512; int hist_h = 400;
	  int bin_w = cvRound( (double) hist_w/histSize );

	  Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	  /// Normalize the result to [ 0, histImage.rows ]
	  normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	  normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
	  normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	  /// Draw for each channel
	  for( int i = 1; i < histSize; i++ )
	  {
	      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
	                       Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
	                       Scalar( 255, 0, 0), 2, 8, 0  );
	      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
	                       Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
	                       Scalar( 0, 255, 0), 2, 8, 0  );
	      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
	                       Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
	                       Scalar( 0, 0, 255), 2, 8, 0  );
	  }

	  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	  imshow("calcHist Demo", histImage );
	  waitKey(0);

}


void image_show_hist_grey(Mat input){

    // Initialize parameters
    int histSize = 256;    // bin size
    float range[] = { 0, 255 };
    const float *ranges[] = { range };

    // Calculate histogram
    MatND hist;
    calcHist( &input, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );

    // Show the calculated histogram in command window
    double total;
    total = input.rows * input.cols;
    for( int h = 0; h < histSize; h++ )
         {
            float binVal = hist.at<float>(h);
            cout<<" "<<binVal;
         }

    // Plot the histogram
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    for( int i = 1; i < histSize; i++ )
    {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
    }

    namedWindow( "Result", 1 );
    imshow( "Result", histImage );
    waitKey(0);
}

