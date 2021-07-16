package celllab;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import javax.imageio.ImageIO;

import java.io.File;

//import java.io.File; // for exporting image
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.analysis.watershed.Component;
import org.openimaj.image.analysis.watershed.feature.MomentFeature;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.colour.Transforms;
import org.openimaj.image.connectedcomponent.GreyscaleConnectedComponentLabeler;
import org.openimaj.image.feature.local.detector.mser.MSERFeatureGenerator;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.ConnectedComponent.ConnectMode;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.convolution.FSobel;
import org.openimaj.image.processing.convolution.FSobelX;
import org.openimaj.image.processing.convolution.FSobelY;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.processing.edges.EdgeFinder;
import org.openimaj.image.processing.edges.NonMaximumSuppressionTangent;
import org.openimaj.image.processing.edges.StrokeWidthTransform;
import org.openimaj.image.processor.PixelProcessor;
import org.openimaj.image.processor.connectedcomponent.render.BlobRenderer;
import org.openimaj.image.processor.connectedcomponent.render.BorderRenderer;
import org.openimaj.image.processor.connectedcomponent.render.OrientatedBoundingBoxRenderer;
import org.openimaj.image.renderer.MBFImageRenderer;
import org.openimaj.image.segmentation.FelzenszwalbHuttenlocherSegmenter;
import org.openimaj.image.segmentation.SegmentationUtilities;
import org.openimaj.math.geometry.shape.Circle;
import org.openimaj.math.geometry.shape.Shape;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;

import javassist.expr.NewArray;

public class KMeans {

	protected List<ConnectedComponent> segment(MBFImage image, String image_name, File pipeline_path,
			boolean plot_pipeline) throws IOException {

		String this_image = "";

		MBFImage image_copy = image.clone();
		
		if (plot_pipeline) {
			this_image = "original_" + image_name;
			ImageUtilities.write(image, new File(pipeline_path, this_image));
		}

		// background subtraction + colour space transformation
		PreProcessor pre = new PreProcessor();
		image = pre.illumination_correction(image);

		if (plot_pipeline) {
			this_image = "alive_illumination_" + image_name;
			ImageUtilities.write(image, new File(pipeline_path, this_image));
		}

		image = ColourSpace.convert(image, ColourSpace.CIE_Lab);

		if (plot_pipeline) {
			this_image = "alive_cielab_" + image_name;
			ImageUtilities.write(image, new File(pipeline_path, this_image));
		}

		// construct K-Means algorithm
		FloatKMeans cluster = FloatKMeans.createExact(15); // 3 or >4< Colours!!! 5 works better for sticky cells

		// flatten and group pixels
		float[][] image_data = image.getPixelVectorNative(new float[image.getWidth() * image.getHeight()][3]);
		FloatCentroidsResult result = cluster.cluster(image_data);

		final float[][] centroids = result.centroids;

		// hard assigner for the image
		final HardAssigner<float[], ?, ?> assigner = result.defaultHardAssigner();

		if (plot_pipeline) {

			MBFImage image_kmeans_copy = image.clone();
			// the pixel processor automatically loops through the image.
			image_kmeans_copy.processInplace(new PixelProcessor<Float[]>() {
				public Float[] processPixel(Float[] pixel) {
					float[] fp = new float[3];
					for (int lab = 0; lab < 3; lab++) {
						fp[lab] = pixel[lab];
					}
					final int centroid = assigner.assign(fp);
					pixel[0] = centroids[centroid][0];
					pixel[1] = centroids[centroid][1];
					pixel[2] = centroids[centroid][2];
					return pixel;
				}
			});
			this_image = "alive_cluster_all_cielab_" + image_name;
			ImageUtilities.write(image_kmeans_copy, new File(pipeline_path, this_image));
			image_kmeans_copy = ColourSpace.convert(image_kmeans_copy, ColourSpace.RGB);
			this_image = "alive_cluster_all_rgb_" + image_name;
			ImageUtilities.write(image_kmeans_copy, new File(pipeline_path, this_image));
		}

		// the pixel processor automatically loops through the image.
		image.processInplace(new PixelProcessor<Float[]>() {
			public Float[] processPixel(Float[] pixel) {
				float[] fp = new float[3];
				for (int lab = 0; lab < 3; lab++) {
					fp[lab] = pixel[lab];
				}
				final int centroid = assigner.assign(fp);

				if (centroids[centroid][1] > 3 || (centroids[centroid][1] > 0 && centroids[centroid][0] > 55)
						|| centroids[centroid][0] > 60) {// )/thr) { // || centroids[centroid][0] <= thr2) {// > 30.0 &&
															// centroids[centroid][lab] < 50.0) {
					pixel[0] = (float) 100.0;
					pixel[1] = (float) 0.51;
					pixel[2] = (float) -0.37;
				} else {
					pixel[0] = (float) 0.0;
					pixel[1] = (float) 0.0;
					pixel[2] = (float) 0.0;
				}

				return pixel;
			}
		});

		image = ColourSpace.convert(image, ColourSpace.RGB);

		if (plot_pipeline) {
			this_image = "alive_cluster_2_rgb_" + image_name;
			ImageUtilities.write(image, new File(pipeline_path, this_image));
		}

		GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
		List<ConnectedComponent> segments = labeler.findComponents(image.flatten());
		
		PostProcessor post = new PostProcessor();
		
		if (plot_pipeline) {
			this_image = "pip_morph_unfiltered_" + image_name;
			MaskExporter exporter = new MaskExporter();
			
			List<ConnectedComponent> seg = post.filter_segments(segments, 0, 18000);
			exporter.save_cells_single_mask(this_image, image_copy.clone(), seg, "alive", pipeline_path);
		}

		
		List<ConnectedComponent> segments_filtered = post.filter_segments(segments, 50, 7000);

		List<ConnectedComponent> segments_poly = new ArrayList<ConnectedComponent>();
		for (ConnectedComponent segment : segments_filtered) {
			ConnectedComponent cc = new ConnectedComponent(segment.toPolygon());
			segments_poly.add(cc);
		}

		return segments_poly;

	}

}