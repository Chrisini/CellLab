package celllab;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.math.geometry.shape.Polygon;

public class CellLab {
	
	// clustering
	void run_kmeans() throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images_type\\alive\\");
		final File pipeline_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\pipeline_java_kmeans");
		final File result_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\java_kmeans");
		
		KMeans segmenter = new KMeans();
		for (final File image_file : path.listFiles()) {
			
			// filename for output
			String image_name = image_file.getName();
			if (image_name.contains("cell15.")) {
			
			// read images
			MBFImage image = ImageUtilities.readMBF(image_file);
			
			// segment regions of interest
			List<ConnectedComponent> segments_filtered = segmenter.segment(image, image_name, pipeline_path, true);
			
			// mask export
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask(image_name, image, segments_filtered, "alive", pipeline_path);
			exporter.save_cells_single_mask_border(image_name, image, segments_filtered, "pip_2", pipeline_path);
//			exporter.save_cells_multiple_masks(image_name, image, segments_filtered, "alive", result_path);
		}
		}
	}
	
	// edge + cht
	void run_edge_cht() throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images_type\\dead\\");
		final File pipeline_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\pipeline_java_cht");
		final File result_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\java_cht");

		CircleHough segmenter = new CircleHough();
		for (final File image_file : path.listFiles()) {
			
			// filename for output
			String image_name = image_file.getName();
			
			if(image_name.contains("cell47")) {
			
			// read images
			MBFImage image = ImageUtilities.readMBF(image_file);
			FImage image_edge = ImageUtilities.readF(image_file);
			
			// segment regions of interest
			List<ConnectedComponent> segments_filtered = segmenter.segment(image, image_edge, image_name, pipeline_path, true);
			
			// mask export
			MaskExporter exporter = new MaskExporter();
//			exporter.save_cells_single_mask(image_name, image, segments_filtered, "pip_cht", pipeline_path);
			exporter.save_cells_single_mask_border(image_name, image, segments_filtered, "pip_cht_AAAA", pipeline_path);
//			exporter.save_cells_multiple_masks(image_name, image, segments_filtered, "dead", result_path);
			}
		}
	}
	
	// edge + morph
	void run_edge_morph() throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images_type\\inhib\\");
		final File pipeline_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\pipeline_java_morph");
		final File result_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\java_morph");

		Morph segmenter = new Morph();
		for (final File image_file : path.listFiles()) {
			
			// filename for output
			String image_name = image_file.getName();		
			if(image_name.contains("cell76")) {
						
			// read images
			MBFImage image = ImageUtilities.readMBF(image_file);
			FImage image_edge = ImageUtilities.readF(image_file);
			
			// segment regions of interest
			List<ConnectedComponent> segments_filtered = segmenter.segment(image, image_edge, image_name, pipeline_path, true);
			
			// mask export
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask(image_name, image, segments_filtered, "inhib", pipeline_path);
//			exporter.save_cells_single_mask_border(image_name, image, segments_filtered, "pip_morph_D", pipeline_path);
//			exporter.save_cells_multiple_masks(image_name, image, segments_filtered, "inhib", result_path);
			}
		}
	}
	
	// graph
	void run_graph() throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images_type\\inhib\\");
		final File pipeline_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\pipeline_java_graph");
		final File result_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\java_graph");
		
		GraphBased segmenter = new GraphBased();
		for (final File image_file : path.listFiles()) {
			
			// filename for output
			String image_name = image_file.getName();
			
			if (image_name.contains("cell76")) {
			
			// read images
			MBFImage image = ImageUtilities.readMBF(image_file);
			
			// segment regions of interest
			List<ConnectedComponent> segments_filtered = segmenter.segment(image, image_name, pipeline_path, true);
			
			// mask export
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask(image_name, image, segments_filtered, "inhib", pipeline_path);
			exporter.save_cells_single_mask_border(image_name, image, segments_filtered, "pip_graph_1150", pipeline_path);
//			exporter.save_cells_multiple_masks(image_name, image, segments_filtered, "inhib", result_path);
			}
		}
	}
	
	// threshold
	void run_threshold() throws IOException {
		
		final File path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images_type\\fibre\\");
		final File pipeline_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\pipeline_java_fibre");
		final File result_path = new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\java_fibre");

		Thresholding segmenter = new Thresholding();
		for (final File image_file : path.listFiles()) {
			
			// filename for output
			String image_name = image_file.getName();
			
			if(image_name.contains("cell18")) {

			// read images
			MBFImage image = ImageUtilities.readMBF(image_file);
			FImage image_thresh = ImageUtilities.readF(image_file);

			// segment regions of interest
			List<ConnectedComponent> segments_filtered = segmenter.segment(image, image_thresh, image_name, pipeline_path, true);

			// mask export
			MaskExporter exporter = new MaskExporter();
			exporter.save_cells_single_mask(image_name, image, segments_filtered, "pip_thresh_dot35", pipeline_path);
//			exporter.save_cells_single_mask_border(image_name, image, segments_filtered, "pip_thresh_dot35", pipeline_path);
//			exporter.save_cells_multiple_masks(image_name, image, segments_filtered, "fibre", result_path);
			}
		}
	}
	

	public static void main(String[] args) throws IOException {
		
		CellLab segmenter = new CellLab();
//		segmenter.run_kmeans();
//		segmenter.run_edge_cht();
		segmenter.run_edge_morph();
//		segmenter.run_graph();
//		segmenter.run_threshold();
		
		System.out.println("Finished");

	}

}
