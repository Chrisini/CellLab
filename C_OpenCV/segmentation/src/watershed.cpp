/*
 * watershed.cpp
 *
 *  Created on: 17 Feb 2020
 *      Author: Prinzessin
 */

/*
 * A pretty good solution is to use morphological closing to make the brightness uniform and then use a regular (non-adaptive) Otsu threshold:
 * 	Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE,
 new Size(19, 19));
 Mat closed = new Mat();
 Imgproc.morphologyEx(image, closed, Imgproc.MORPH_CLOSE, kernel);

 */

#include "../inc/segmentation.hpp"
using namespace std;

using namespace cv;

Mat watershed_do(Mat input) {

	Mat output;

	return output;

}

Mat distance_and_watershed(Mat src) {

	// Show source image
	imshow("Source Image", src);

	Mat lala;

	lala = noise_gauss_blur_correction(src);
	lala = noise_reduce(lala);

	imshow("lala", lala);

	// Create a kernel that we will use to sharpen our image
	Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1); // an approximation of second derivative, a quite strong kernel
	// do the laplacian filtering as it is
	// well, we need to convert everything in something more deeper then CV_8U
	// because the kernel has some negative values,
	// and we can expect in general to have a Laplacian image with negative values
	// BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
	// so the possible negative number will be truncated
	Mat imgLaplacian;
	filter2D(src, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	src.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	// imshow( "Laplace Filtered Image", imgLaplacian );
	imshow("New Sharped Image", imgResult);
	// Create binary image from source image
	Mat bw;
	/*  cvtColor(imgResult, bw, COLOR_BGR2GRAY);
	 threshold(bw, bw, 40, 255, THRESH_BINARY | THRESH_OTSU);
	 imshow("Binary Image", bw);
	 */

	Mat output, output2;
	Mat original;

	output = image_read();
	output2 = image_read();

	output = noise_reduce_colour(imgResult);
	output = feature_contour(output, 150);    // CONTOUR_REDUCE_VALUE);

	output2 = noise_gauss_blur_correction(output2);
	output2 = feature_contour(output2, 100);

	imshow("Contour", output);

	Mat im_fill, im_fill_inverse;
	bw = colour_grey(output);

	imshow("BW1", bw);

	threshold(bw, bw, 1, 255, THRESH_BINARY);
	/*	im_fill = bw.clone();
	 floodFill(im_fill, cv::Point(0,0), Scalar(255));
	 bitwise_not(im_fill, im_fill_inverse);
	 bw = (bw | im_fill_inverse);
	 imshow("BW", bw);
	 */

	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, DIST_L2, 3);
	// Normalize the distance image for range = {0.0, 1.0}
	// so we can visualize and threshold it
	normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	imshow("Distance Transform Image", dist);
	// Threshold to obtain the peaks
	// This will be the markers for the foreground objects
	threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(5, 5, CV_8U); // 3,3
	dilate(dist, dist, kernel1);
	floodFill(dist, cv::Point(0, 0), Scalar(255));
	imshow("Peaks", dist);
	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
	vector<vector<Point> > contours;
	findContours(bw, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(bw.size(), CV_32S);
	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++) {
		drawContours(markers, contours, static_cast<int>(i),
				Scalar(static_cast<int>(i) + 1), -1);
	}
	// Draw the background marker
	circle(markers, Point(5, 5), 3, Scalar(255), -1);
	imshow("Markers", markers * 10000);
	// Perform the watershed algorithm
	watershed(lala, markers); // imgResult, markers
	Mat mark;
	markers.convertTo(mark, CV_8U);
	bitwise_not(mark, mark);
	//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
	// image looks like at that point
	// Generate random colors
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		int b = theRNG().uniform(0, 256);
		int g = theRNG().uniform(0, 256);
		int r = theRNG().uniform(0, 256);
		colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
	}
	// Create the result image
	Mat dst = Mat::zeros(markers.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++) {
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size())) {
				src.at<Vec3b>(i, j) = colors[index - 1]; //dst
			}
		}
	}

	/*
	 // https://answers.opencv.org/question/120698/drawning-labeling-components-in-a-image-opencv-c/?sort=oldest
	 Mat stats, centroids, labelImage;
	 int nLabels = connectedComponentsWithStats(mark, labelImage, stats, centroids, 8, CV_32S);
	 Mat mas(labelImage.size(), CV_8UC1, Scalar(0));
	 Mat surfSup = stats.col(4)> 5; //particleval

	 for (int i = 1; i < nLabels; i++)
	 {
	 if (surfSup.at<uchar>(i, 0))
	 {
	 mas = mas | (labelImage==i);
	 }
	 }

	 int nObjects = connectedComponentsWithStats(mas, labelImage, stats, centroids, 8, CV_32S);
	 cout << "Number of objects:" << nObjects  << endl;

	 Mat r(markers.size(), CV_8UC1, Scalar(0));
	 src.copyTo(r,mas);
	 imshow("Mask", mas);
	 imshow("Overlay", r);
	 waitKey();

	 */

	// Visualize the final image
	imshow("Final Result", dst);
	waitKey();
	return dst;
}

Mat distance_and_watershed2(Mat src) {

	// Show source image
	imshow("Source Image", src);

	// Blurred image
	Mat blurred_src;
	blurred_src = noise_gauss_blur_correction(src);
	blurred_src = noise_reduce(blurred_src);
	imshow("blurred source image", blurred_src);

	// laplacian image
	Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
	Mat imgLaplacian;
	filter2D(src, imgLaplacian, CV_32F, kernel);
	Mat sharp;
	src.convertTo(sharp, CV_32F);
	Mat imgResult = sharp - imgLaplacian;
	// convert back to 8bits gray scale
	imgResult.convertTo(imgResult, CV_8UC3);
	imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	// imshow( "Laplace Filtered Image", imgLaplacian );
	imshow("New Sharped Image", imgResult);

	Mat binary;
	Mat output = src.clone();
//	output = noise_reduce_colour(imgResult);
//	output = feature_contour(output, 150);// CONTOUR_REDUCE_VALUE);

	output = noise_gauss_blur_correction(output);

	threshold(output, output, 40, 200, THRESH_BINARY); // 40 for big images, 10 for small ones

	binary = colour_grey(output);
	imshow("binary image", binary);

	/*
	 // Perform the distance transform algorithm
	 Mat dist;
	 distanceTransform(binary, dist, DIST_L2, 3);
	 // Normalize the distance image for range = {0.0, 1.0}
	 // so we can visualize and threshold it
	 normalize(dist, dist, 0, 1.0, NORM_MINMAX);
	 imshow("Distance Transform Image", dist);
	 // Threshold to obtain the peaks
	 // This will be the markers for the foreground objects
	 threshold(dist, dist, 0.4, 1.0, THRESH_BINARY);
	 // Dilate a bit the dist image
	 Mat kernel1 = Mat::ones(5, 5, CV_8U); // 3,3
	 dilate(dist, dist, kernel1);
	 floodFill(dist, cv::Point(0,0), Scalar(255));
	 imshow("Peaks", dist);
	 // Create the CV_8U version of the distance image
	 // It is needed for findContours()
	 Mat dist_8u;
	 dist.convertTo(dist_8u, CV_8U);
	 */

	// Find total markers
	vector<vector<Point> > contours;
	findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	// Create the marker image for the watershed algorithm
	Mat markers = Mat::zeros(binary.size(), CV_32S);
	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++) {
		if (contours.at(i).size() > 10)
			drawContours(markers, contours, static_cast<int>(i),
					Scalar(static_cast<int>(i) + 1), -1);
	}
	imshow("Markers1", markers * 100); //*10000);
	// Draw the background marker
	circle(markers, Point(5, 5), 3, Scalar(255), -1);
	imshow("Markers", markers * 10000);
	// Perform the watershed algorithm
	watershed(blurred_src, markers); // imgResult, markers
	imshow("after watershed", blurred_src);
	Mat mark;
	markers.convertTo(mark, CV_8U);
	bitwise_not(mark, mark);
	//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
	// image looks like at that point
	// Generate random colors
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++) {
		int b = theRNG().uniform(0, 256);
		int g = theRNG().uniform(0, 256);
		int r = theRNG().uniform(0, 256);
		colors.push_back(Vec3b((uchar) b, (uchar) g, (uchar) r));
	}
	// Create the result image
	Mat dst = Mat::zeros(src.size(), CV_8UC3);
	// Fill labeled objects with random colors
	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++) {
			int index = markers.at<int>(i, j);
			if (index > 0 && index <= static_cast<int>(contours.size())) {
				dst.at<Vec3b>(i, j) = colors[index - 1];
			}
		}
	}

	/*

	 */

	// https://answers.opencv.org/question/120698/drawning-labeling-components-in-a-image-opencv-c/?sort=oldest
	Mat stats, centroids, labelImage;
	int nLabels = connectedComponentsWithStats(mark, labelImage, stats,
			centroids, 8, CV_32S);
	Mat mas(labelImage.size(), CV_8UC1, Scalar(0));
	Mat surfSup = stats.col(4) > 5; //particleval

	for (int i = 1; i < nLabels; i++) {
		if (surfSup.at<uchar>(i, 0)) {
			mas = mas | (labelImage == i);
		}
	}

	int nObjects = connectedComponentsWithStats(mas, labelImage, stats,
			centroids, 8, CV_32S);
	cout << "Number of objects:" << nObjects << endl;

	Mat r(markers.size(), CV_8UC1, Scalar(0));
	src.copyTo(r, mas);
	imshow("Mask", mas);
	imshow("Overlay", r);
	waitKey();

	// Visualize the final image
	imshow("Final Result", dst);
	waitKey();
	return dst;
}
