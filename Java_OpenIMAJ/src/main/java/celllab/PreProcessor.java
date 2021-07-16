package celllab;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.histogram.HistogramAnalyser;
import org.openimaj.image.pixel.statistics.HistogramModel;
import org.openimaj.image.processing.algorithm.EqualisationProcessor;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.math.statistics.distribution.Histogram;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;


public class PreProcessor {

	/* ********** Returns a single-colour image that will be used as corrected background ********** */
	private MBFImage mean_colour_image(MBFImage image) throws IOException {
		// clone image
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

	/* ********** Gets rid of illuminated background and replaces it with single-coloured background ********** */
	protected MBFImage illumination_correction(MBFImage original_image) throws MalformedURLException, IOException {

		// Read original image and blur
		MBFImage gauss_image = original_image.clone(); //ImageUtilities.readMBF(imgFile);
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
//		DisplayUtilities.display(original_image);
//		DisplayUtilities.display(gauss_image);
//		DisplayUtilities.display(mean_gauss_image);
//		DisplayUtilities.display(corrected_image);
		
		return corrected_image;

	}

	public static void main(String[] args) throws MalformedURLException, IOException {
		// Path to image
		final File imgFile = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images\\cell57.jpg");
		
		// image
		MBFImage original_image = ImageUtilities.readMBF(imgFile);
		
		// New object of this class
		PreProcessor preproc = new PreProcessor();

		// Illumination correction + background subtraction		
		original_image = preproc.illumination_correction(original_image);
		
		// Normalisation
//		original_image = original_image.normalise();
		
		// Histogram Equalisation
		original_image.processInplace(new EqualisationProcessor());
		DisplayUtilities.display(original_image);
		
		ImageUtilities.write(original_image, new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\preproc\\illum_equ.jpg"));

	}
}
