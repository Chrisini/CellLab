package uk.ac.soton.ecs.ceb1;

import org.apache.hadoop.classification.InterfaceAudience.Public;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

import org.openimaj.image.FImage;
import org.openimaj.image.processor.SinglebandImageProcessor;

/**
 * 
 * Resources:
 */
public class MyConvolution implements SinglebandImageProcessor<Float, FImage> {
	private float[][] kernel;

	public MyConvolution(float[][] kernel) {
		// note that like the image pixels kernel is indexed by [row][column]
		this.kernel = kernel; // template
	}

	/*
	 * Convolution via averaging operator / box filter
	 */
	@Override
	public void processImage(FImage image) {
		// convolve image with kernel and store result back in image
		//
		// hint: use FImage#internalAssign(FImage) to set the contents
		// of your temporary buffer image to the image.

		/*
		 * (1) Create destination image with size of source image (2) Set all pixels to
		 * black, so border will remain with no information
		 */
		FImage destination_image = new FImage(image.width, image.height);
		for (int row = 0; row <= image.height; row++) {
			for (int col = 0; col <= image.width; col++) {
				destination_image.setPixel(row, col, (float) 0.0);
			}
		}

		/*
		 * Get every pixel, without borders
		 */
		int middle_kernel_row = (kernel.length - 1) / 2; // starting from index 0
		int middle_kernel_col = (kernel[0].length - 1) / 2;
		float value = 0;
		for (int row = middle_kernel_row; row < image.height - middle_kernel_row; row++) {
			for (int col = middle_kernel_col; col < image.width - middle_kernel_col; col++) {
				value = apply_convolution(image, row, col);
				destination_image.setPixel(row+middle_kernel_row, col+middle_kernel_col, value);
			}
		}
		
		

		/*
		 * 
		 * float kernel_width = image. + kernel.length-1; // row float kernel_height =
		 * image. + kernel[0].length-1; // column
		 * 
		 * 
		 * float[][] multi = new float[kernel[0].length][kernel.length]; FImage
		 * destination_image = new FImage(multi);
		 */
		// destination_image.
	}

	/*
	 * Calculates
	 */
	public float apply_convolution(FImage image, int start_row, int start_col) {
		float center_value = 0;
		int row = 0, col = 0;

		for (row = 0; row < kernel.length - 1; row++) {
			for (col = 0; col <= kernel[0].length - 1; col++) {
				center_value += image.getPixel(start_row + row, start_col + col) * kernel[row][col];
			}
		}
		return center_value * (-1);
	}

	/*
	 * Black corners for kernel with size 3x3
	 */
	public void black_edges(FImage image) {

		// int middle_kernel_row = (kernel.length-1)/2; // starting from index 0
		// int middle_kernel_col = (kernel[0].length-1)/2;
		int border_row = image.height; // - middle_kernel_row;
		int border_col = image.width; // - middle_kernel_col;

		for (int col = 0; col < border_col; col++) {
			image.pixels[0][col] = (float) 0.0;
			image.pixels[border_row][col] = (float) 0.0;
		}

		for (int row = 0; row < border_row; row++) {
			image.pixels[row][0] = (float) 0.0;
			image.pixels[row][border_col] = (float) 0.0;
		}
	}

}