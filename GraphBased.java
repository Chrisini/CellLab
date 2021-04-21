package celllab;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.SingleBandImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.ConnectedComponent.ConnectMode;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.morphology.Close;
import org.openimaj.image.processing.morphology.Dilate;
import org.openimaj.image.processing.morphology.Erode;
import org.openimaj.image.processing.morphology.StructuringElement;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processor.connectedcomponent.render.BlobRenderer;
import org.openimaj.image.processor.connectedcomponent.render.BorderRenderer;
import org.openimaj.image.processor.connectedcomponent.render.ConfigurableRenderOptions;
import org.openimaj.image.processor.connectedcomponent.render.OrientatedBoundingBoxRenderer;
import org.openimaj.image.saliency.AchantaSaliency;
import org.openimaj.image.segmentation.FelzenszwalbHuttenlocherSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;
import org.openimaj.math.geometry.point.PointList;
import org.openimaj.math.geometry.shape.Polygon;


/**
 * Felzenszwalb Huttenlocher Segmenter
 * 
 * 
 * Note, when using polygon, we can simplify shape with reduce vertices
// Polygon poly = segment.toPolygon();
// eps - distance value below which a vertex can be ignored
// poly = poly.reduceVertices(1);
// p_list.add(poly);
 * 					
 */

public class GraphBased {

	
	MBFImage mean_colour_image(MBFImage image) throws IOException {
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

	/*
	 * Gets rid of illuminated background and replaces it with single-coloured background
	 */
	MBFImage illumination_correction(final File imgFile) throws MalformedURLException, IOException {
		// Read image
		MBFImage original_image = ImageUtilities.readMBF(imgFile);

		// Read original image and blur
		MBFImage gauss_image = ImageUtilities.readMBF(imgFile);
		gauss_image.processInplace(new FGaussianConvolve(10f));

		// Create empty image and get mean of blurred image
		MBFImage mean_gauss_image = new MBFImage();
		mean_gauss_image = mean_colour_image(gauss_image);

		// Create empty image, subtract blurred image from original image and add the
		// mean of the blurred image
		MBFImage corrected_image = new MBFImage();
		corrected_image = original_image.subtract(gauss_image);
		corrected_image = corrected_image.add(mean_gauss_image);

		return corrected_image;
	}
	
	private void save_mask(String image_name, MBFImage image, MBFImage border, List<ConnectedComponent> filtered_segments) throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\computer_masks\\graph_java2");
		
		int i = 0;
		
		for (final PixelSet segment : filtered_segments) {
			
			MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
			mask.fill(RGBColour.BLACK);
			
			final BlobRenderer<Float[]> br = new BlobRenderer<Float[]>(mask, RGBColour.WHITE); //, ConnectMode.CONNECT_8);
		    br.process(new ConnectedComponent(segment.pixels));
		    
		    String this_image = "dead_" + Integer.toString(i) + "_" + image_name;
		    ImageUtilities.write(mask, new File(path, this_image));
		    i++;
		    
		}
		
		System.out.println("Done with file " + image_name);
		
	}
	
	private void graphbased_algorithm() throws MalformedURLException, IOException {
		
		
		//final File dir = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\annotation_images");
		final File dir = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images");
		for(final File image_file : dir.listFiles()) {
			
			String file_name = image_file.getName();
			// Read image
			// final File imgFile = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images\\cell35.jpg");
			MBFImage image = new MBFImage();
			image = illumination_correction(image_file);
			// image = ImageUtilities.readMBF(imgFile);
			
			// Create borders
			MBFImage border = new MBFImage();
			border = ImageUtilities.readMBF(image_file);
			
			// Create mask
			MBFImage mask = new MBFImage();
			mask = ImageUtilities.readMBF(image_file);
	
			// Normalise image
			//image = image.normalise();
			//DisplayUtilities.display(image);
						
			// Create segmenter
			FelzenszwalbHuttenlocherSegmenter<MBFImage> fhs = new FelzenszwalbHuttenlocherSegmenter<MBFImage>(); // 0.5f, 100f/50f, 50
	
			// Put segments (connected components) into list
			List<ConnectedComponent> segments = fhs.segment(image);
			// List for segments with the right size
			List<ConnectedComponent> filtered_segments = new ArrayList<>();
			
			List<Polygon> p_list = new ArrayList<>();
			
			// Filter segments with wrong size			
			for (ConnectedComponent segment : segments) {
				if(segment.calculateArea() > 100 && segment.calculateArea() < 10000) {
					
					Close c = new Close(StructuringElement.CROSS);
					c.process(segment);
					
					filtered_segments.add(segment);
				}
			}
			
			// Render segment
			image = SegmentationUtilities.renderSegments(image, filtered_segments); //image.getWidth(), image.getHeight()
	
			// Create blue bounding boxes
			OrientatedBoundingBoxRenderer<Float[]> obbr = new OrientatedBoundingBoxRenderer<Float[]>(mask, RGBColour.BLUE);
			ConnectedComponent.process(filtered_segments, obbr);
			
			//save_mask(file_name, image, border, filtered_segments);
			
			//  RGBColour.randomColour()
			// BlobRenderer, BorderRenderer
			for (final PixelSet segment : filtered_segments) {
				final BlobRenderer<Float[]> br = new BlobRenderer<Float[]>(border, RGBColour.WHITE); //, ConnectMode.CONNECT_8);
			    br.process(new ConnectedComponent(segment.pixels));
			}
	
			// Export image as jpg file
	//		final File path = new File("C:\\Users\\Prinzessin\\Pictures");
	//		ImageUtilities.write(image, new File(path, "chrisy8.jpg"));
	
			//ImageUtilities.write(original, new File("D:\\felz_dead_blue.jpg"));
			// DisplayUtilities.display(mask);
			//DisplayUtilities.display(border);
		}
	}
	
	public static void main(String[] args) throws MalformedURLException, IOException{
		
		 GraphBased segmenter = new GraphBased();
		 segmenter.graphbased_algorithm();
		
	}

}
