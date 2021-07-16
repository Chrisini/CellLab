package celllab;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.contour.Contour;
import org.openimaj.image.contour.ContourRenderer;
import org.openimaj.image.contour.SuzukiContourProcessor;
import org.openimaj.image.contour.SuzukiNeighborStrategy;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.processing.morphology.Close;
import org.openimaj.image.processing.morphology.Dilate;
import org.openimaj.image.processing.morphology.Erode;
import org.openimaj.image.processing.morphology.Open;
import org.openimaj.image.processing.morphology.StructuringElement;
import org.openimaj.image.processing.morphology.Thicken;
import org.openimaj.image.processing.threshold.OtsuThreshold;
import org.openimaj.math.geometry.shape.Polygon;

public class Morph {
	
	protected List<ConnectedComponent> segment(MBFImage image, FImage image_edge, String image_name,
			File pipeline_path, boolean plot_pipeline) throws IOException {

		String this_image = "";
		
		if (plot_pipeline) {
			this_image = "original_" + image_name;
			ImageUtilities.write(image, new File(pipeline_path, this_image));
		}
		
		if (plot_pipeline) {
			this_image = "pip_morph_grey_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		
		image_edge = image_edge.normalise();
		
		if (plot_pipeline) {
			this_image = "pip_morph_norm_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		
		FImage copy = image_edge.clone();
		copy.processInplace(new FGaussianConvolve(50f));
		image_edge = image_edge.subtract(copy);
		
		if (plot_pipeline) {
			this_image = "pip_morph_subtract_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		
		// edges
		CannyEdgeDetector canny = new CannyEdgeDetector((float) 0.01, (float) 0.15, (float) 0.5);
		canny.processImage(image_edge);		
		
		if (plot_pipeline) {
			this_image = "pip_morph_edges_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		
		image_edge = image_edge.process(new Dilate(StructuringElement.CROSS));
		if (plot_pipeline) {
			this_image = "pip_morph_1dilate_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		image_edge = image_edge.process(new Close(StructuringElement.CROSS));
		if (plot_pipeline) {
			this_image = "pip_morph_2close_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		image_edge = image_edge.process(new Dilate(StructuringElement.CROSS));
		if (plot_pipeline) {
			this_image = "pip_morph_3dilate_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		image_edge = image_edge.process(new Close(StructuringElement.CROSS));
		if (plot_pipeline) {
			this_image = "pip_morph_4close_" + image_name;
			ImageUtilities.write(image_edge, new File(pipeline_path, this_image));
		}
		
		
		GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
		List<ConnectedComponent> segments = labeler.findComponents(image_edge);
		PostProcessor post = new PostProcessor();
		System.out.println(segments);
		segments = post.filter_segments_close(segments, 0, 190000);
		System.out.println(segments);
		
		if (plot_pipeline) {
			this_image = "pip_morph_comp_" + image_name;
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask(this_image, image.clone(), segments, "inhib", pipeline_path);
		}

		// postprocessor - filter segments that overlap
		//PostProcessor post = new PostProcessor();
		List<ConnectedComponent> segments_filtered = post.filter_segments_close(segments, 500, 60000);

		if (plot_pipeline) {
			this_image = "pip_morph_filtered_" + image_name;
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask(this_image, image.clone(), segments_filtered, "inhib", pipeline_path);
		}
		
		List<ConnectedComponent> segments_poly = new ArrayList<ConnectedComponent>();
		for(ConnectedComponent segment : segments_filtered) {
			ConnectedComponent cc = new ConnectedComponent(segment.toPolygon());
			segments_poly.add(cc);
		}
		
		if (plot_pipeline) {
			this_image = "pip_morph_poly_" + image_name;
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask(this_image, image.clone(), segments_poly, "inhib", pipeline_path);
		}
		
		return segments_poly;

	}
	
}
