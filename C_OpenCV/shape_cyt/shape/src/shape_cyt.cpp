/*
 * canny_edge_detection.cpp
 *
 *  Created on: 29.09.2019
 *      Author: Prinzessin
 */

/* make
 * ./DisplayImage data/zellen_tot_rand.jpg
 */

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

Mat image, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThresh;
int const max_lowThresh = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge map";

void CannyThresh(int, void*){
	blur(src_gray, detected_edges, Size(3,3));
	Canny(detected_edges, detected_edges, lowThresh, lowThresh*ratio, kernel_size);

	dst = Scalar::all(0);

	image.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    dst.create(image.size(), image.type());
    cvtColor(image, src_gray, CV_BGR2GRAY);
    namedWindow(window_name, WINDOW_NORMAL);

    createTrackbar("Min: ", window_name, &lowThresh, max_lowThresh, CannyThresh);

    CannyThresh(0,0);

    waitKey(0);
    return 0;
}
