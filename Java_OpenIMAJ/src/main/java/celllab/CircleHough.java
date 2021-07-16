package celllab;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.math.geometry.shape.Shape;

public class CircleHough {

	protected List<ConnectedComponent> segment(MBFImage image, FImage image_edge, String image_name,
			File pipeline_path, boolean plot_pipeline) throws IOException {

		String this_image = "";
		
		if (plot_pipeline) {
			this_image = "original_" + image_name;
			ImageUtilities.write(image, new File(pipeline_path, this_image));
		}
		
		if (plot_pipeline) {
			this_image = "grey_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		
		
		image_edge = image_edge.normalise();
		
		if (plot_pipeline) {
			this_image = "dead_norm_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		
		// edges
		CannyEdgeDetector canny = new CannyEdgeDetector((float) 0.01, (float) 0.2, (float) 2); //(float) 0.01, (float) 0.2, 1
		canny.processImage(image_edge);

		if (plot_pipeline) {
			this_image = "dead_edge_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}

		// hough circles
		HoughCircles circles = new HoughCircles(7, 15, 1, 80);
		circles.analyseImage(image_edge);

		// get best 300 circles
		List<HoughCircles.WeightedCircle> segments = circles.getBest(500);
		
		// before filtering, clone image to not overwrite anything
		if (plot_pipeline) {
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask_border(image_name, image.clone(), segments, "dead_unfiltered_", pipeline_path, true);
		}

		// postprocessor - filter segments that overlap
		PostProcessor post = new PostProcessor();
		List<HoughCircles.WeightedCircle> segments_filtered = post.filter_segments_cht(segments);
		List<ConnectedComponent> segments_cc = new ArrayList<ConnectedComponent>();
		
		for (Shape segment : segments_filtered) {
			ConnectedComponent cc = new ConnectedComponent(segment);
			segments_cc.add(cc);
		}

		return segments_cc;

	}

}
