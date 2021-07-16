package uk.ac.soton.ecs.ceb1n19.ch4;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.statistics.HistogramModel;
import org.openimaj.math.statistics.distribution.MultidimensionalHistogram;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done: true
 * 
 * @task: 4.1.1. Exercise 1: Finding and displaying similar images 4.1.2.
 *        Exercise 2: Exploring comparison measures
 *        
 * @resources The OpenIMAJ Tutorial
 */
public class CompareImages {

	double distance = 0.0;
	int id1 = 0, id2 = 1;

	/**
	 * Intersection: The images with the highest number match the most 1.0 is the
	 * most similar one - here we got 1.000002 and 0.99 as well Again, the two
	 * zebras were the two most similar ones
	 * 
	 * Histogram intersection s(H1,H2) = sumI( min(H1(I), H2(I)) )
	 * 
	 * @param histograms
	 */
	private void compare_intersection(List<MultidimensionalHistogram> histograms) {
		double highest = 0.0;
		for (int i = 0; i < histograms.size(); i++) {
			for (int j = i; j < histograms.size(); j++) {
				distance = histograms.get(i).compare(histograms.get(j), DoubleFVComparison.INTERSECTION);
				if (distance < 0.9 && highest <= distance) {
					highest = distance;
					id1 = i;
					id2 = j;
				}
			}
		}
	}

	/**
	 * Euclidean: Compare histograms with each other and save the IDs from the two
	 * most similar ones, which are not the same (0.0) The two zebras were the most
	 * similar images, as expected
	 * 
	 * Euclidean distance d(H1,H2) = Math.sqrt( sumI( (H1(I)-H2(I))^2 ) )
	 * 
	 * @param histograms
	 */
	private void compare_euclidean(List<MultidimensionalHistogram> histograms) {
		double lowest = 1.0;
		for (int i = 0; i < histograms.size(); i++) {
			for (int j = i; j < histograms.size(); j++) {
				distance = histograms.get(i).compare(histograms.get(j), DoubleFVComparison.EUCLIDEAN);
				if (distance != 0.0 && lowest >= distance) {
					lowest = distance;
					id1 = i;
					id2 = j;
				}
			}
		}
	}

	/**
	 * Load images from URL
	 * 
	 * @return
	 * @throws MalformedURLException
	 */
	private URL[] load_images() throws MalformedURLException {
		URL[] image_urls = new URL[] { new URL(
				"https://images2.minutemediacdn.com/image/upload/c_fill,g_auto,h_1248,w_2220/f_auto,q_auto,w_1100/v1554997857/shape/mentalfloss/istock-177363117.jpg"),
				new URL("https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fjamiecartereurope%2Ffiles%2F2019%2F02%2F1-2-1200x675.jpg"),
				new URL("https://i.pinimg.com/originals/24/01/2d/24012dd2fb30c0b530176b6c8c4dad30.jpg"),
				new URL("http://www.italien-region-marken.de/wp-content/uploads/2017/01/Zebra.jpg"),
				new URL("https://cdn10.bigcommerce.com/s-x8dfmo/products/8700/images/31754/Thunderbird-2-in-Thunderbirds-Premium-Photograph-and-Poster-1015146__92732.1432428503.1280.1280.jpg?c=2") };
		return image_urls;
	}

	/**
	 * Display the two images with the biggest similarity
	 * 
	 * @param image_urls
	 * @throws IOException
	 */
	private void display_similar(URL[] image_urls) throws IOException {

		// Create images form the IDs we got from comparing the histograms
		MBFImage image = new MBFImage();
		image = ImageUtilities.readMBF(image_urls[id1]);
		MBFImage image2 = new MBFImage();
		image2 = ImageUtilities.readMBF(image_urls[id2]);

		DisplayUtilities.display(image);
		DisplayUtilities.display(image2);
	}

	/**
	 * Create histograms and save in list
	 * 
	 * @param image_urls
	 * @return
	 * @throws IOException
	 */
	private static List<MultidimensionalHistogram> create_histogram_list(URL[] image_urls) throws IOException {
		List<MultidimensionalHistogram> histograms = new ArrayList<MultidimensionalHistogram>();
		HistogramModel model = new HistogramModel(4, 4, 4);

		for (URL u : image_urls) {
			model.estimateModel(ImageUtilities.readMBF(u));
			histograms.add(model.histogram.clone());
		}

		return histograms;
	}

	public static void main(String[] args) throws MalformedURLException, IOException {
		CompareImages im = new CompareImages();
		URL[] urls = im.load_images();
		List<MultidimensionalHistogram> histograms = create_histogram_list(urls);
		im.compare_intersection(histograms);
		im.compare_euclidean(histograms);
		im.display_similar(urls);
	}

}
