package uk.ac.soton.ecs.ceb1n19.ch1;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.general.GeneralFont;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done true
 * 
 * @task 1.2.1. Exercise 1: Playing with the sample application
 * 
 * @resources The OpenIMAJ Tutorial
 */
public class GoodMorningWorld {

	/**
	 * An image is created by text
	 */
	private void create_image_from_text() {
		// Create an image
		MBFImage image = new MBFImage(580, 70, ColourSpace.RGB);

		// Fill the image with white
		image.fill(RGBColour.LIGHT_GRAY);

		// Create new general fonts
		// GeneralFont lato_font = new GeneralFont("Lato", 1);
		// GeneralFont calibri_font = new GeneralFont("Calibri", 1);
		GeneralFont verdana_font = new GeneralFont("Verdana", 5);

		// Render some test into the image
		image.drawText("Good Morning World!", 10, 60, verdana_font, 50, RGBColour.RGB(12, 218, 240));

		// Apply a Gaussian blur
		image.processInplace(new FGaussianConvolve(2f));

		// Display the image
		DisplayUtilities.display(image);
	}

	public static void main(String[] args) {
		GoodMorningWorld gmw = new GoodMorningWorld();
		gmw.create_image_from_text();
	}
}
