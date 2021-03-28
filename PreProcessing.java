package celllab;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.FGaussianConvolve;


public class PreProcessing {

	/*
	 * Returns a single-colour image that will be used as corrected background
	 */
	MBFImage mean_colour_image(MBFImage image) throws IOException {

		MBFImage clone = image.clone();

		float red = 0;
		float blue = 0;
		float green = 0;

		int counter = 0;

		// go through each pixel and save value
		for (int width = 0; width < image.getWidth()-1; width++) {
			for (int height = 0; height < image.getHeight()-1; height++) {
				float[] channels = image.getPixelNative(width, height);
				red += channels[0];
				green += channels[1];
				blue += channels[2];
				counter++;
			}
		}

		// calculate mean value for each channel
		red = red / counter;
		green = green / counter;
		blue = blue / counter;

		// overwrite each pixel with the mean value
		for (int width = 0; width < image.getWidth(); width++) {
			for (int height = 0; height < image.getHeight(); height++) {
				clone.getBand(0).setPixel(width, height, red);
				clone.getBand(1).setPixel(width, height, green);
				clone.getBand(2).setPixel(width, height, blue);
			}
		}
		return clone;
	}

	/*
	 * Gets rid of illuminated background and replaces it with single-coloured background
	 */
	void illumination_correction(final File imgFile) throws MalformedURLException, IOException {

		// Read image
		MBFImage original_image = ImageUtilities.readMBF(imgFile);

		// Read original image and blur
		MBFImage gauss_image = ImageUtilities.readMBF(imgFile);
		gauss_image.processInplace(new FGaussianConvolve(50f));

		// Create empty image and get mean of blurred image
		MBFImage mean_gauss_image = new MBFImage();
		mean_gauss_image = mean_colour_image(gauss_image);

		// Create empty image, subtract blurred image from original image and add the
		// mean of the blurred image
		MBFImage corrected_image = new MBFImage();
		corrected_image = original_image.subtract(gauss_image);
		corrected_image = corrected_image.add(mean_gauss_image);

		// Show images
		DisplayUtilities.display(original_image);
		DisplayUtilities.display(gauss_image);
		DisplayUtilities.display(mean_gauss_image);
		DisplayUtilities.display(corrected_image);

	}

	/*
	 * Normalise an image
	 */
	void normalise_image(final File imgFile) throws MalformedURLException, IOException {
		// Read image
		MBFImage image = new MBFImage();
		image = ImageUtilities.readMBF(imgFile);

		// Normalise image
		image = image.normalise();
		
		// Show image
		DisplayUtilities.display(image);
	}

	public static void main(String[] args) throws MalformedURLException, IOException {
		// TODO Auto-generated method stub

		// Path to image
		final File imgFile = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images\\cell57.jpg");
		
		// New object of this class
		PreProcessing preproc = new PreProcessing();

		// Illumination correction / background subtraction
		preproc.illumination_correction(imgFile);
		
		// Normalisation
		preproc.normalise_image(imgFile);

	}
}
