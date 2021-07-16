package celllab;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.processing.morphology.Close;
import org.openimaj.image.processing.morphology.StructuringElement;
import org.openimaj.image.segmentation.FelzenszwalbHuttenlocherSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;


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
	
	protected List<ConnectedComponent> segment(MBFImage image, String image_name, File pipeline_path, boolean plot_pipeline) throws IOException {

			String this_image = "";
		
			if (plot_pipeline) {
				this_image = "original_" + image_name;
				ImageUtilities.write(image, new File(pipeline_path, this_image));
			}
			
			// preprocessor - correct illumination
			PreProcessor pre = new PreProcessor();
			image = pre.illumination_correction(image);
			
			if (plot_pipeline) {
				this_image = "correction" + image_name;
				ImageUtilities.write(image, new File(pipeline_path, this_image));
			}
						
			// create segmenter
			FelzenszwalbHuttenlocherSegmenter<MBFImage> fhs = new FelzenszwalbHuttenlocherSegmenter<MBFImage>(0.5f, 2f, 100); // (0.5f, 100f/50f, 50)
	
			// put segments (connected components) into list
			List<ConnectedComponent> segments = fhs.segment(image);
						
			// postprocessor - filter segments with wrong size
			PostProcessor post = new PostProcessor();
			List<ConnectedComponent> segments_filtered = post.filter_segments_close(segments, 100, 10000);
			
			return segments_filtered;
	}

}
