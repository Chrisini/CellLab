/*
 * segmentation.hpp
 *
 *  Created on: 11 Feb 2020
 *      Author: Prinzessin
 */

#ifndef INC_SEGMENTATION_HPP_
#define INC_SEGMENTATION_HPP_

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;


Mat algorithm_fibre(Mat image, int threshnumber, int particle_size);
Mat algorithm_kmeans(Mat original);
Mat algorithm_cht(Mat original);

Mat image_read();
void image_show(Mat image);

Mat watershed_do(Mat image);
Mat distance_and_watershed(Mat input);
Mat distance_and_watershed2(Mat src);


Mat cell_crop(Mat img, Rect roi);

Mat apply_fibre(Mat fibres);

Rect roi_getter(Mat img, int offset_x, int offset_y);

Mat noise_reduce_colour(Mat img);
Mat noise_reduce(Mat img);
Mat noise_blur(Mat img);
Mat noise_gauss_blur_correction(Mat input);

Mat colour_grey(Mat input);

Mat kmeans(Mat image);

Mat threshold_adaptive(Mat input);

Mat feature_canny_edge(Mat input);
Mat feature_contour(Mat input, int threshnumber, int particle_size = 15);
//Mat feature_contour_fibre(Mat input, int threshnumber, int particle_size = 15);

Mat morph_erode(Mat input);
Mat morph_dilate(Mat input);
Mat morph_open(Mat input);
Mat morph_close(Mat input);
Mat morph_fill(Mat input);

void image_show_hist(Mat input);
void image_show_hist_grey(Mat input);

#define CONTOUR_PSEUDO_VALUE 30
#define CONTOUR_REDUCE_VALUE 160


#define NUM_CLUSTERS 5



#endif /* INC_SEGMENTATION_HPP_ */
