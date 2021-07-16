#include "inc/segmentation.hpp"
#include "inc/cell.hpp"
using namespace cv;
using namespace std;

Mat original;
int threshval = 100;
int particleval = 5;

#define CHOICE 4

void segment() {

	Mat output;
	Mat original;
	int position[2]; // will get 2 dimensional eventually?
	Rect roi;
	Mat water;

	output = image_read();
	output = noise_gauss_blur_correction(output);
	image_show(output);
	output = colour_grey(output);
	image_show_hist_grey(output);
//	original = image_read();
//	original = noise_reduce_colour(original);
//	original = feature_canny_edge(original);
	//original = morph_dilate(original);
	//original = morph_open(original);
	//output = noise_reduce_colour(output);
	output = feature_contour(output, 20);

	//water = do_watershed(image);
//	roi = get_roi(output, 100, 200);
//
//	//position = get_position(image);
//	//roi = get_roi(image, position);
//
//	output = crop_cell(output, roi);
//
//	output = reduce_colour_noise(output);
//	output = canny_edge_feature(output);

	// output = crop_cell(original, x, y); // position

	image_show(output);
}

void segment_contour() {

	Mat output, output2;
	Mat original;

	output = image_read();
	output2 = image_read();
	original = image_read();

	/*	output = noise_reduce_colour(output);
	 imshow("Noise reduction", output);
	 output = feature_contour(output, 150);	// CONTOUR_REDUCE_VALUE);
	 */
	// out = cv2.normalize(new_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	Mat img, gray, float_gray, blur, num, den;

	// Load color image

	// convert to grayscale
	cv::cvtColor(output2, gray, CV_BGR2GRAY);

	// convert to floating-point image
	gray.convertTo(float_gray, CV_32F, 1.0 / 255.0);

	// numerator = img - gauss_blur(img)
	cv::GaussianBlur(float_gray, blur, Size(0, 0), 2, 2);
	num = float_gray - blur;

	imshow("num", num);

	// denominator = sqrt(gauss_blur(img^2))
	cv::GaussianBlur(num.mul(num), blur, Size(0, 0), 20, 20);
	cv::pow(blur, 0.5, den);

	imshow("den", den);

	// output = numerator / denominator
	gray = num / den;

	imshow("grey", gray);

	// normalize output into [0,1]
	cv::normalize(gray, gray, 0.0, 1.0, NORM_MINMAX, -1);

	// Display
	namedWindow("demo", CV_WINDOW_AUTOSIZE);
	imshow("demo", gray);

	float Min = 0;
	float Max = 1;
	gray.convertTo(output2, CV_8U, 255.0 / (Max - Min),
			-255.0 * Min / (Max - Min));

	output2 = noise_blur(output2);
	output2 = noise_blur(output2);
	output2 = noise_blur(output2);
	output2 = noise_blur(output2);
	imshow("Noise reduced 2", output2);
	output2 = noise_gauss_blur_correction(output2);
	//output2 = colour_grey(output2);
	//equalizeHist(output2, output2);
	//normalize(gray, output2, 100, 50);
	// output2 = noise_reduce(output2);
	imshow("Gaussian Blur", output2);
	output2 = feature_contour(output2, 10);

	float alpha = 0.6;
	float beta = 0.3;
	beta = (1.0 - alpha);
	addWeighted(original, alpha, output2, beta, 0.0, output2);

//	imshow("Contour", output);

	/*	Mat im_fill, im_fill_inverse;
	 output = colour_grey(output);
	 threshold(output, output, 100, 255, THRESH_BINARY);
	 im_fill = output.clone();
	 floodFill(im_fill, cv::Point(0,0), Scalar(255));
	 bitwise_not(im_fill, im_fill_inverse);
	 output = (output | im_fill_inverse);*/

	namedWindow("Contour: Reduce Colour", CV_WINDOW_AUTOSIZE);
	imshow("Contour: Reduce Colour", output);
	namedWindow("Contour: Gauss Correction", CV_WINDOW_AUTOSIZE);
	imshow("Contour: Gauss Correction", output2);
	waitKey(0);
}

//https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
void segment_morph() {
	Mat output, detected_edges, dst;

	output = image_read();
	// TODO: for bachelor thesis, compare with and without correction
	output = noise_gauss_blur_correction(output);

	output = colour_grey(output);

	//equalizeHist( output, output );

	// image_show_hist_grey(output);

	output = noise_reduce(output);
	output = noise_blur(output);

	// output = threshold_adaptive(output);
	threshold(output, output, 50, 255, THRESH_OTSU);

	output = morph_erode(output);
	output = morph_dilate(output);
	output = morph_open(output);
	output = morph_close(output);

	blur(output, detected_edges, Size(3, 3));
	Canny(detected_edges, detected_edges, 0, 0 * 3, 3);
	dst = Scalar::all(0);
	output.copyTo(dst, detected_edges);

	namedWindow("Morphology", CV_WINDOW_AUTOSIZE);
	imshow("Morphology", output);
	waitKey(0);

}

void segment_watershed() {

	Mat image;
	image = image_read();
	distance_and_watershed2(image);
}

/**
 * Create folder for each image
 * Save mask for each cell of the image as:
 * round | chicken | hug | fibre
 *
 */
void save_masks(int choice, int fibre, Mat original) {

	int number = 0;

	cout << number << " of masks will be saved" << endl;

}

/**
 *
 */
void choose_masks(Mat four_choices[4], Mat fibres, Mat original) {

	int choice = 0;
	int fibre = 0;

	resize(original, original, Size(320, 240));

	int i = 0;
	for (i = 0; i < CHOICE; i++) {
		resize(four_choices[i], four_choices[i], Size(320, 240));
	}

//	fibres.convertTo(fibres, CV_8UC3, 255.0);
	resize(fibres, fibres, Size(320, 240));

	// Create 1280x480 mat for window
	cv::Mat win_mat(cv::Size(1920, 240), CV_8UC3);

	// f small images into big mat
	original.copyTo(win_mat(cv::Rect(0, 0, 320, 240)));
	four_choices[0].copyTo(win_mat(cv::Rect(320, 0, 320, 240)));
	four_choices[1].copyTo(win_mat(cv::Rect(640, 0, 320, 240)));
	four_choices[2].copyTo(win_mat(cv::Rect(960, 0, 320, 240)));
	four_choices[3].copyTo(win_mat(cv::Rect(1280, 0, 320, 240)));
	fibres.copyTo(win_mat(cv::Rect(1600, 0, 320, 240)));

	// Display big mat
//	namedWindow("Images", CV_WINDOW_AUTOSIZE);
	// namedWindow("Images", CV_WINDOW_AUTOSIZE);
	imshow("Images", win_mat);
	waitKey(500);

	cout << "Choose a mask (0) - none, (1), (2), (3), (4)" << endl;
	cin >> choice;

	cout << "Are fibres detected correctly? 1 = yes, 2 = no" << endl;
	cin >> fibre;

	save_masks(choice, fibre, original);

}

/**
 * 4 algorithms
 * + fibre
 */
void do_algorithms(Mat original) {

	Mat four_choices[4];

	Mat fibres;

	four_choices[0] = original.clone();
//	four_choices[0] = apply_watershed(four_choices[0]);

	four_choices[1] = original.clone();
//	four_choices[1] = apply_contours(four_choices[1]);

	four_choices[2] = original.clone();
//	four_choices[2] = apply_kmeans(four_choices[2]);

	four_choices[3] = original.clone();
//	four_choices[3] = apply_hough(four_choices[3]);

	fibres = original.clone();
	fibres = apply_fibre(fibres);

	choose_masks(four_choices, fibres, original);

}

// https://answers.opencv.org/question/69712/load-multiple-images-from-a-single-folder/
void read_images() {

	String path = "data/celllab_train/all/*";
	vector<String> fn;
	vector<Mat> images;

	// GET NUMBER OF OBJECTS IN FOLDER
	glob(path, fn, false);

	// READ ALL IMAGES
	for (size_t k = 0; k < fn.size(); ++k) {
		cv::Mat im = cv::imread(fn[k]);
		if (im.empty())
			continue; //only proceed if sucsessful

		cout << "*********" << fn[k] << "*********" << endl;

		do_algorithms(im);

		images.push_back(im);

	}
}

static void on_trackbar(int, void*) {

	// Fibres
	algorithm_fibre(original, threshval, particleval);

	// k-Means
	algorithm_kmeans(original);

	// watershed
//	distance_and_watershed2(original);

	// CHT
	algorithm_cht(original);
//    Mat labelImage(img.size(), CV_32S);
//    int nLabels = connectedComponents(bw, labelImage, 8);
//    std::vector<Vec3b> colors(nLabels);
//    colors[0] = Vec3b(0, 0, 0);//background
//    for(int label = 1; label < nLabels; ++label){
//        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
//    }
//
//    // showing connected components of bw
//    Mat dst(img.size(), CV_8UC3);
//    for(int r = 0; r < dst.rows; ++r){
//        for(int c = 0; c < dst.cols; ++c){
//            int label = labelImage.at<int>(r, c);
//            Vec3b &pixel = dst.at<Vec3b>(r, c);
//            pixel = colors[label];
//         }
//     }
//    imshow( "Connected Components", dst );

	// showing contours of bw

//    Mat labels;
//	Mat stats;
//	Mat centroids;
//	cv::connectedComponentsWithStats(bw, labels, stats, centroids);

//	cout << labels << endl;
//	cout << centroids << endl;
//	cout << stats << endl;

}

int main(int argc, char **argv) {

	original = imread("data/celllab_train/all/cell104.jpg", 1);
	if (original.empty()) {
		cout << "Could not read input image file:";
		return EXIT_FAILURE;
	}
	resize(original, original, Size(320, 240));

	imshow("Original image", original);
	namedWindow("Images", WINDOW_AUTOSIZE);
	createTrackbar("Threshold (Fibre)", "Images", &threshval, 255, on_trackbar);
	createTrackbar("Particle Size (Fibre)", "Images", &particleval, 255,
			on_trackbar);
	on_trackbar(threshval, 0);
	on_trackbar(particleval, 0);
	waitKey(0);

//	read_images();

	//segment();
	//segment_morph();
	//segment_contour();
	//segment_watershed();

	destroyAllWindows();

	return 0;
}
