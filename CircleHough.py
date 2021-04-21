package celllab;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.processing.edges.EdgeFinder;
import org.openimaj.image.processing.edges.NonMaximumSuppression;
import org.openimaj.image.processing.edges.SUSANEdgeDetector;
import org.openimaj.image.renderer.MBFImageRenderer;
import org.openimaj.math.geometry.shape.Shape;

import org.openimaj.image.processing.convolution.FSobel;

/*
 * This only works good if objects and background have a high contrast.
 */

public class CircleHough {

	private void save_pipeline_image(String image_name, MBFImage image, String type) throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\disseration_images\\pipeline_cht_java");
		
		DisplayUtilities.display(image);

		String this_image = "cht_java_" + type + "_" + image_name;
	    ImageUtilities.write(image, new File(path, this_image));
	}
	
	private void save_pipeline_image(String image_name, FImage image, String type) throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\disseration_images\\pipeline_cht_java");
		
		
		DisplayUtilities.display(image, "pipeline");

		String this_image = "cht_java_" + type + "_" + image_name;
	    //ImageUtilities.write(image, new File(path, this_image));
	}
	
	private void save_mask(String image_name, MBFImage image, List<HoughCircles.WeightedCircle> list_circles_filtered)
			throws IOException {

		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\computer_masks\\cht_java");

		MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
		
		int i = 0;

		for (final Shape cir : list_circles_filtered) {
			mask.fill(RGBColour.BLACK);
			final MBFImageRenderer renderer = mask.createRenderer();
			renderer.drawShapeFilled(cir, RGBColour.WHITE);

			String this_image = "dead_" + Integer.toString(i) + "_" + image_name;
			ImageUtilities.write(mask, new File(path, this_image));
			i++;
		}

		System.out.println("Done with file " + image_name);

	}
	
	FImage mean_colour_image(FImage image) throws IOException {
		// clone image
		FImage clone = image.clone();

		float red = 0;
		float blue = 0;
		float green = 0;

		int counter = 0;

		// go through each pixel and save value
		for (int width = 0; width < image.getWidth()-1; width++) {
			for (int height = 0; height < image.getHeight()-1; height++) {
				red += image.getPixelNative(width, height);
				counter++;
			}
		}

		// calculate mean value for each channel
		red = red / counter;

		// overwrite each pixel with the mean value
		for (int width = 0; width < image.getWidth(); width++) {
			for (int height = 0; height < image.getHeight(); height++) {
				clone.setPixel(width, height, red);
			}
		}
		return clone;
	}
	
	FImage illumination_correction(FImage imgFile) throws MalformedURLException, IOException {
		// Read image
		FImage original_image = imgFile.clone();

		// Read original image and blur
		FImage gauss_image = imgFile.clone();
		gauss_image.processInplace(new FGaussianConvolve(10f));

		// Create empty image and get mean of blurred image
		FImage mean_gauss_image = mean_colour_image(gauss_image);

		// Create empty image, subtract blurred image from original image and add the
		// mean of the blurred image
		FImage corrected_image = original_image.subtract(gauss_image);
		corrected_image = corrected_image.add(mean_gauss_image);

		return corrected_image;
	}

	private void hough_algorithm() throws IOException {

		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images_type\\dead\\");
		
		for(final File image_file : path.listFiles()) {

		String image_name = image_file.getName();
			
		FImage image_edge = ImageUtilities.readF(image_file);
		MBFImage image_circles = ImageUtilities.readMBF(image_file);
		MBFImage image_circles_filtered = ImageUtilities.readMBF(image_file);
		MBFImage image_mask = new MBFImage(image_circles.getWidth(), image_circles.getHeight(), 3);
		image_mask.fill(RGBColour.BLACK);

		//image_edge = illumination_correction(image_edge);
		//DisplayUtilities.display(image_edge); 
				
		FSobel sob = new FSobel(3f);
		sob.analyseImage(image_edge);
		
		CannyEdgeDetector ced = new CannyEdgeDetector();
		ced.processImage(image_edge, sob);

		HoughCircles circles = new HoughCircles(7, 15, 1, 80);
		circles.analyseImage(image_edge);

		List<HoughCircles.WeightedCircle> list_circles = circles.getBest(50);
		List<HoughCircles.WeightedCircle> list_circles_filtered = circles.getBest(50);

		// remove circles that overlap (-3)
		for (int this_cell = 0; this_cell < list_circles.size(); this_cell++) {
			float this_x = list_circles.get(this_cell).getX();
			float this_y = list_circles.get(this_cell).getY();
			float this_radi = list_circles.get(this_cell).getRadius();

			for (int other_cell = this_cell + 1; other_cell < list_circles.size(); other_cell++) {
				float other_x = list_circles.get(other_cell).getX();
				float other_y = list_circles.get(other_cell).getY();
				float other_radi = list_circles.get(other_cell).getRadius();

				int x_dist = (int) this_x - (int) other_x;
				int y_dist = (int) this_y - (int) other_y;

				if (Math.sqrt(x_dist * x_dist + y_dist * y_dist) < (this_radi + other_radi - 3)) {
					// remove circle with smaller radius
					if (other_radi < this_radi) {
						list_circles_filtered.remove(list_circles.get(other_cell));
					} else {
						list_circles_filtered.remove(list_circles.get(this_cell));
					}
				}
			}
		}

		for (final Shape cir : list_circles) {
			final MBFImageRenderer renderer = image_circles.createRenderer();
			renderer.drawShape(cir, 2, RGBColour.BLACK);
		}

		for (final Shape cir : list_circles_filtered) {
			final MBFImageRenderer renderer = image_circles_filtered.createRenderer();
			renderer.drawShape(cir, 2, RGBColour.BLACK);
		}
		
		for (final Shape cir : list_circles_filtered) {
			final MBFImageRenderer renderer = image_mask.createRenderer();
			renderer.drawShapeFilled(cir, RGBColour.WHITE);
		}

		save_mask(image_name, image_circles, list_circles_filtered);

		save_pipeline_image(image_name, image_edge, "edge");
		// save_pipeline_image(image_name, image_circles, "circles");
		save_pipeline_image(image_name, image_circles_filtered, "filtered");
		// save_pipeline_image(image_name, image_mask, "mask");	

		}
	}

	public static void main(String[] args) throws IOException {

		CircleHough circle = new CircleHough();
		circle.hough_algorithm();
	}

}
