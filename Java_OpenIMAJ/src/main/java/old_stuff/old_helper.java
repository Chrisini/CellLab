package old_stuff;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.renderer.MBFImageRenderer;
import org.openimaj.math.geometry.shape.Shape;

public class old_helper {
	
	public static void main(String[] args) {
		private void save_pipeline_image(String image_name, MBFImage image, String type) throws IOException {
			
			final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\disseration_images\\pipeline_cht_java");
			
			DisplayUtilities.display(image);

			String this_image = "cht_java_" + type + "_" + image_name;
//		    ImageUtilities.write(image, new File(path, this_image));
		}
		
		private void save_pipeline_image(String image_name, FImage image, String type) throws IOException {
			
			final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\disseration_images\\pipeline_cht_java");
			
			DisplayUtilities.display(image, "pipeline");

			String this_image = "cht_java_" + type + "_" + image_name;
//		    ImageUtilities.write(image, new File(path, this_image));
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
		
		
//		List<ConnectedComponent> segments_filtered_cc = new ArrayList<ConnectedComponent>();
//		for (final Shape cir : segments) {
//			final MBFImageRenderer renderer = image_demo_circles.createRenderer();
//			renderer.drawShape(cir, 2, RGBColour.BLACK);
//		}

//		for (final Shape cir : segments_filtered) {
//			final MBFImageRenderer renderer = image_demo_circles_filtered.createRenderer();
//			renderer.drawShape(cir, 2, RGBColour.BLACK);
//		}
//		
//		for (final Shape cir : segments_filtered) {
//			final MBFImageRenderer renderer = image_mask.createRenderer();
//			renderer.drawShapeFilled(cir, RGBColour.WHITE);
//		}

//		save_mask(image_name, image_demo_circles, segments_filtered);

//		save_pipeline_image(image_name, image_edge, "edge");
//		save_pipeline_image(image_name, image_circles, "circles");
//		save_pipeline_image(image_name, image_demo_circles_filtered, "filtered");
//		save_pipeline_image(image_name, image_mask, "mask");	
		
	}
	
	
	
	

}
