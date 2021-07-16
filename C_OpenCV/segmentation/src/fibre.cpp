/*
 * fibre.cpp
 *
 *  Created on: 14 Jun 2020
 *      Author: Prinzessin
 */


#include "../inc/segmentation.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>
#include "math.h"
#include <ctime>
using namespace cv;
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <vector>

/**
 *	algorithm to detect fibres
 *	(1) Thresholding
 *	(2) Morphology operators
 *	(3) Connected Components
 *	(4) Get rid of objects smaller than particleval
 *	Todo: Mask out of contours
 */
Mat algorithm_fibre(Mat original, int threshval, int particleval) {

	RNG rng(12345);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat mask;
	Mat fibres = original.clone();

	blur(fibres, fibres, Size(3, 3));
	if (fibres.channels() != 1)
		cvtColor(fibres, fibres, CV_BGR2GRAY);
	fibres = threshval < 128 ? (fibres < threshval) : (fibres > threshval);

	int morph_size = 1;
	Mat element = getStructuringElement(MORPH_ELLIPSE,
			Size(2 * morph_size + 1, 2 * morph_size + 1),
			Point(morph_size, morph_size));
	morphologyEx(fibres, fibres, MORPH_CLOSE, element);

	// https://answers.opencv.org/question/120698/drawning-labeling-components-in-a-image-opencv-c/?sort=oldest
    Mat stats, centroids, labelImage;
    int nLabels = connectedComponentsWithStats(fibres, labelImage, stats, centroids, 8, CV_32S);
    Mat mas(labelImage.size(), CV_8UC1, Scalar(0));
    Mat surfSup = stats.col(4)>particleval;

    for (int i = 1; i < nLabels; i++)
    {
        if (surfSup.at<uchar>(i, 0))
        {
            mas = mas | (labelImage==i);
        }
    }

    int nObjects = connectedComponentsWithStats(mas, labelImage, stats, centroids, 8, CV_32S);
    cout << "Number of objects:" << nObjects  << endl;

    Mat r(original.size(), CV_8UC1, Scalar(0));
     original.copyTo(r,mas);
    imshow("Mask", mas);
    imshow("Overlay", r);
    waitKey();

	return mas;
}

// not used I think ...
Mat apply_fibre(Mat fibres){

	fibres = imread( "data/celllab_train/cell111.bmp", 1);
			//cell_1.bmp
			// colours.png

	resize(fibres, fibres, Size(640, 480));

	Mat mask;

	// mask = kmeans(fibres);
	//mask = feature_contour_fibre(mask, 20, 100);

	imshow("Chrisy",mask);

	return mask;

}
