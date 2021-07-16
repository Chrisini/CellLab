package uk.ac.soton.ecs.ceb1n19.ch2;

import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.typography.hershey.HersheyFont;
import org.openimaj.math.geometry.shape.Ellipse;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done true
 * 
 * @tasks 2.1.1. Exercise 1: DisplayUtilities 2.1.2. Exercise 2: Drawing
 * 
 * @resources The OpenIMAJ Tutorial
 */
public class Images {

	private void draw_images() {

		// Display exactly one image at a time
		DisplayUtilities.createNamedWindow("image_window", "Chapter 2", true);

		// Create images from file and URL
		MBFImage image_from_file = new MBFImage();
		MBFImage image_from_url = new MBFImage();
		try {
			image_from_file = ImageUtilities.readMBF(new File("C:\\Users\\Prinzessin\\Pictures\\file.jpg")); // "file.jpg"
			image_from_url = ImageUtilities.readMBF(new URL(
					"https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fjamiecartereurope%2Ffiles%2F2019%2F02%2F1-2-1200x675.jpg"));
		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// Print colour space, in this case RGB
		System.out.println(image_from_file.colourSpace);

		// Display images
		// DisplayUtilities.display(image_from_file, "File");
		// DisplayUtilities.display(image_from_file, "URL");
		DisplayUtilities.displayName(image_from_file, "image_window");
		DisplayUtilities.displayName(image_from_url, "image_window");

		// Display gray image, red channel
		// DisplayUtilities.display(image_from_url.getBand(0), "Gray");
		DisplayUtilities.displayName(image_from_url.getBand(0), "image_window");

		// Clone image and set blue and green channel to zero
		MBFImage red_image = image_from_url.clone();
		red_image.getBand(1).fill(0f);
		red_image.getBand(2).fill(0f);
		// DisplayUtilities.display(red_image, "Red");
		DisplayUtilities.displayName(red_image, "image_window");

		// Display image with Canny Edge Filter
		red_image.processInplace(new CannyEdgeDetector());
		// DisplayUtilities.display(red_image, "Canny");
		DisplayUtilities.displayName(red_image, "image_window");

		// Draw a speech bubble on an image
		image_from_url.drawShape(new Ellipse(550f, 450f, 20f, 10f, 0f), 5, RGBColour.MAGENTA);
		image_from_url.drawShapeFilled(new Ellipse(550f, 450f, 20f, 10f, 0f), RGBColour.GREEN);
		image_from_url.drawShape(new Ellipse(500f, 425f, 25f, 12f, 0f), 5, RGBColour.MAGENTA);
		image_from_url.drawShapeFilled(new Ellipse(500f, 425f, 25f, 12f, 0f), RGBColour.GRAY);
		image_from_url.drawShape(new Ellipse(450f, 380f, 30f, 15f, 0f), 5, RGBColour.MAGENTA);
		image_from_url.drawShapeFilled(new Ellipse(450f, 380f, 30f, 15f, 0f), RGBColour.WHITE);
		image_from_url.drawShape(new Ellipse(350f, 300f, 100f, 70f, 0f), 5, RGBColour.MAGENTA);
		image_from_url.drawShapeFilled(new Ellipse(350f, 300f, 100f, 70f, 0f), RGBColour.WHITE);
		image_from_url.drawText("OpenIMAJ is", 275, 300, HersheyFont.ASTROLOGY, 20, RGBColour.BLUE);
		image_from_url.drawText("Awesome", 275, 330, HersheyFont.ASTROLOGY, 20, RGBColour.CYAN);
		// DisplayUtilities.display(image_from_url, "Drawing");
		DisplayUtilities.displayName(image_from_url, "image_window");

	}

	public static void main(String[] args) {
		Images i = new Images();
		i.draw_images();
	}

}
