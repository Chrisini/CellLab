/*
 * kmeans.cpp
 *
 *  Created on: 15 Jun 2020
 *      Author: Prinzessin
 */

#include "../inc/segmentation.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>
#include "math.h"
#include <ctime>
using namespace cv;
using namespace std;

Mat algorithm_kmeans(Mat original) {

	Mat ocv = original.clone();
	int k_value = 8;

	// convert to float & reshape to a [3 x W*H] Mat
	//  (so every pixel is on a row of it's own)
	Mat data;
	ocv.convertTo(data,CV_32F);
	data = data.reshape(1,data.total());

	// do kmeans
	Mat labels, centers;
	kmeans(data, k_value, labels, TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3,
	       KMEANS_PP_CENTERS, centers);

	cout << "center" << centers << endl;

	// reshape both to a single row of Vec3f pixels:
	centers = centers.reshape(3,centers.rows);
	data = data.reshape(3,data.rows);

	// replace pixel values with their center value:
	Vec3f *p = data.ptr<Vec3f>();
	for (size_t i=0; i<data.rows; i++) {
	   int center_id = labels.at<int>(i);
	   p[i] = centers.at<Vec3f>(center_id);
	}

	// back to 2d, and uchar:
	ocv = data.reshape(3, ocv.rows);
	ocv.convertTo(ocv, CV_8U);

	imshow("kmeans2", ocv);

	return ocv;

	/*Mat image, blurred;

	// backgrouns subtraction
	int high = original.rows/2;
	int wide = original.cols/2;
	if (high % 2 == 0) ++ high;
	if (wide % 2 == 0) ++ wide;
	GaussianBlur(original, blurred, Size(wide, high), 0, 0 );
	subtract(original, blurred, image);
	image = noise_gauss_blur_correction(original);

	imshow("blurred", blurred);
	imshow("subtracted image", image);


	cv::Mat kMeansSrc(image.rows * image.cols, 3, CV_32F);
	//resize the image to src.rows*src.cols x 3
	//cv::kmeans expects an image that is in rows with 3 channel columns
	//this rearranges the image into (rows * columns, numChannels)
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			for (int z = 0; z < 3; z++)
				kMeansSrc.at<float>(y + x * image.rows, z) = image.at<Vec3b>(y,
						x)[z];
		}
	}

	cv::Mat labels;
	cv::Mat centers;
	int attempts = 2;
//perform kmeans on kMeansSrc with NUM_CLUSTERS
//end either when desired accuracy is met or the maximum number of iterations is reached
	cv::kmeans(kMeansSrc, NUM_CLUSTERS, labels,
			cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 8, 1),
			attempts, KMEANS_PP_CENTERS, centers);

//create an array of NUM_CLUSTERS colors
	int colors[NUM_CLUSTERS];
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		colors[i] = 255 / (i + 1);
	}

	std::vector<cv::Mat> layers;

	for (int i = 0; i < NUM_CLUSTERS; i++) {
		layers.push_back(cv::Mat::zeros(image.rows, image.cols, CV_32F));
	}

//use the labels to draw the layers
//using the array of colors, draw the pixels onto each label image
	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			int cluster_idx = labels.at<int>(y + x * image.rows, 0);
			layers[cluster_idx].at<float>(y, x) = (float) (colors[cluster_idx]);
		}
	}

	std::vector<cv::Mat> srcLayers;

//each layer to mask a portion of the original image
//this leaves us with sections of similar color from the original image
	for (int i = 0; i < NUM_CLUSTERS; i++) {
		layers[i].convertTo(layers[i], CV_8UC1);
		srcLayers.push_back(cv::Mat());
		image.copyTo(srcLayers[i], layers[i]);
		imshow("Kmeans", srcLayers[i]);
		waitKey(2000);
		cout << i;
	}

	return srcLayers[2];
*/
}
