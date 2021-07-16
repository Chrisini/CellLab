package celllab;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.ConnectedComponent.ConnectMode;
import org.openimaj.image.processing.threshold.AdaptiveLocalThresholdMean;
import org.openimaj.image.processing.threshold.OtsuThreshold;
import org.openimaj.image.segmentation.ConnectedThresholdSegmenter;
import org.openimaj.math.matrix.similarity.processor.Threshold;

public class Thresholding {

	
	protected List<ConnectedComponent> segment(MBFImage image, FImage image_thresh, String image_name,
			File pipeline_path, boolean plot_pipeline) throws IOException {
		
		String this_image = "";
		
		if(plot_pipeline) {
			this_image = "original_" + image_name;
			ImageUtilities.write(image, new File(pipeline_path, this_image));
		}
		
		if(plot_pipeline) {
			this_image = "grey_" + image_name;
			ImageUtilities.write(image_thresh, new File(pipeline_path, this_image));
		}
		
		image_thresh = image_thresh.threshold((float) 0.35);
		
		if(plot_pipeline) {
			this_image = "pip_thresh_" + image_name;
			ImageUtilities.write(image_thresh, new File(pipeline_path, this_image));
		}
		
		image_thresh = image_thresh.inverse();
		
		if(plot_pipeline) {
			this_image = "pip_thresh_inverse_" + image_name;
			ImageUtilities.write(image_thresh, new File(pipeline_path, this_image));
		}
		
		GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
		List<ConnectedComponent> segments = labeler.findComponents(image_thresh);
		System.out.println(segments);
		
		// postprocessor - filter segments with wrong size
		PostProcessor post = new PostProcessor();
		List<ConnectedComponent> segments_filtered = post.filter_segments_close(segments, 500, 60000);
		
//		System.out.println(segments);
		
		return segments_filtered;
	}
}
