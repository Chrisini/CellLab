/*
 * noise.cpp
 *
 *  Created on: 11 Feb 2020
 *      Author: Prinzessin
 */



// fastNlMeansDenoisingColored

#include "../inc/segmentation.hpp"
using namespace cv;


/*
 * Good results: fastNlMeansDenoisingColored(img, output, 3, 3, 7, 21 );
 * this one seems to work best for the cell images
 */
Mat noise_reduce_colour(Mat input){
	Mat output;
	fastNlMeansDenoisingColored(input, output, 3, 3, 7, 21 );
	return output;
}

Mat noise_reduce(Mat input){
	Mat output;
	fastNlMeansDenoising(input, output, 3, 5, 4);
	return output;
}

Mat noise_blur(Mat input){
	Mat output;
	GaussianBlur(input, output, Size( 3, 3 ), 0, 0 );
	return output;
}

/*
 * to get rid of a gradiant & background of an image:
 * take a big kernel for Gaussian-blurring
 * subtract Gaussian-blurred image from the original image
 *
 * int high = input.rows/4;
	int wide = input.cols/4;
	//if (high < 50) high = input.rows/2;
	//if (wide < 50) wide = input.cols/2;
	if (high % 2 == 0) ++ high;
	if (wide % 2 == 0) ++ wide;
	std::cout << high;
	GaussianBlur(input, blurred, Size(wide, high), 0, 0 );
	subtract(input, blurred, output);
	imshow("blur", blurred);
	return output; // needs to be output
*/
Mat noise_gauss_blur_correction(Mat input){
	Mat output, blurred;
	int high = input.rows/4;
	int wide = input.cols/4;
	//if (high < 50) high = input.rows/2;
	//if (wide < 50) wide = input.cols/2;
	if (high % 2 == 0) ++ high;
	if (wide % 2 == 0) ++ wide;
	std::cout << "Filter" << high << "    ";
	GaussianBlur(input, blurred, Size(wide, high), 0, 0 );
	subtract(input, blurred, output);
	imshow("blur", blurred);
	return output; // needs to be output

}
