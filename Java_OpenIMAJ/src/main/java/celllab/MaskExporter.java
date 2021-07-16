package celllab;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.analysis.watershed.WatershedProcessor;
import org.openimaj.image.analysis.watershed.WatershedProcessorAlgorithm;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.ConnectedComponent.ConnectMode;
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processor.connectedcomponent.render.BlobRenderer;
import org.openimaj.image.processor.connectedcomponent.render.BorderRenderer;
import org.openimaj.image.renderer.MBFImageRenderer;
import org.openimaj.math.geometry.shape.Polygon;
import org.openimaj.math.geometry.shape.Shape;


public class MaskExporter {

	/* ********** FILL MULTIPLE MASK USING CC *********** */
	protected void save_cells_multiple_masks(String image_name, MBFImage image,
			List<ConnectedComponent> segments_filtered, String cell_stage, File path_export) throws IOException {

		int i = 0;

		for (final PixelSet segment : segments_filtered) {

			MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
			mask.fill(RGBColour.BLACK);

			final BlobRenderer<Float[]> br = new BlobRenderer<Float[]>(mask, RGBColour.WHITE); // ,
																								// ConnectMode.CONNECT_8);
			br.process(new ConnectedComponent(segment.pixels));

			String this_image = cell_stage + "_" + Integer.toString(i) + "_" + image_name;
			ImageUtilities.write(mask, new File(path_export, this_image));
			i++;

		}

	}

	/* ********** FILL MULTIPLE MASK USING SHAPE *********** */
	protected void save_cells_multiple_masks(String image_name, MBFImage image,
			List<HoughCircles.WeightedCircle> segments_filtered, String cell_stage, File path_export, boolean shape)
			throws IOException {

		int i = 0;

		for (final Shape segment : segments_filtered) {
			MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
			mask.fill(RGBColour.BLACK);

			final MBFImageRenderer renderer = mask.createRenderer();
			renderer.drawShapeFilled(segment, RGBColour.WHITE);

			String this_image = cell_stage + "_" + Integer.toString(i) + "_" + image_name;
			ImageUtilities.write(mask, new File(path_export, this_image));
			i++;
		}

	}

	/* ********** FILL MULTIPLE MASK USING POLY *********** */
	protected void save_cells_multiple_masks_poly(String image_name, MBFImage image,
			List<Polygon> segments_filtered, String cell_stage, File path_export)
			throws IOException {

		int i = 0;

		for (final Shape segment : segments_filtered) {
			MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
			mask.fill(RGBColour.BLACK);

			final MBFImageRenderer renderer = mask.createRenderer();
			renderer.drawShapeFilled(segment, RGBColour.WHITE);

			String this_image = cell_stage + "_" + Integer.toString(i) + "_" + image_name;
			ImageUtilities.write(mask, new File(path_export, this_image));
			i++;
		}

	}
	
	/* ********** FILL SINGLE MASK USING CC *********** */
	protected void save_cells_single_mask(String image_name, MBFImage image, List<ConnectedComponent> segments_filtered,
			String cell_stage, File path_export) throws IOException {

		MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
		mask.fill(RGBColour.BLACK);

		for (final PixelSet segment : segments_filtered) {
			final BlobRenderer<Float[]> br = new BlobRenderer<Float[]>(mask, RGBColour.WHITE); // ,
																								// ConnectMode.CONNECT_8);
			br.process(new ConnectedComponent(segment.pixels));
		}

		String this_image = cell_stage + "_all_" + image_name;
		ImageUtilities.write(mask, new File(path_export, this_image));

	}

	/* ********** FILL SINGLE MASK USING SHAPE *********** */
	protected void save_cells_single_mask(String image_name, MBFImage image,
			List<HoughCircles.WeightedCircle> segments_filtered, String cell_stage, File path_export, boolean shape) // HoughCircles.WeightedCircle
			throws IOException {

		MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
		mask.fill(RGBColour.BLACK);

		for (final Shape segment : segments_filtered) {
			final MBFImageRenderer renderer = mask.createRenderer();
			renderer.drawShapeFilled(segment, RGBColour.WHITE);
		}

		WatershedProcessor wp = new WatershedProcessor();
		wp.processImage(mask.flatten());
		
		DisplayUtilities.display(mask);
		
		String this_image = cell_stage + "_all_" + image_name;
		ImageUtilities.write(mask, new File(path_export, this_image));

	}

	/* ********** FILL SINGLE MASK USING Polygon *********** */
	protected void save_cells_single_mask_poly(String image_name, MBFImage image,
			List<Polygon> segments_filtered, String cell_stage, File path_export) // HoughCircles.WeightedCircle
			throws IOException {

		MBFImage mask = new MBFImage(image.getWidth(), image.getHeight(), 3);
		mask.fill(RGBColour.BLACK);

		for (final Shape segment : segments_filtered) {
			final MBFImageRenderer renderer = mask.createRenderer();
			renderer.drawShapeFilled(segment, RGBColour.WHITE);
		}

		WatershedProcessor wp = new WatershedProcessor();
		wp.processImage(mask.flatten());
		
		DisplayUtilities.display(mask);
		
		String this_image = cell_stage + "_all_" + image_name;
		ImageUtilities.write(mask, new File(path_export, this_image));

	}
	
	/* ********** BORDER SINGLE MASK USING CC *********** */
	protected void save_cells_single_mask_border(String image_name, MBFImage image,
			List<ConnectedComponent> segments_filtered, String cell_stage, File path_export) throws IOException {
		for (final PixelSet segment : segments_filtered) {
			final CellBorderRenderer<Float[]> br = new CellBorderRenderer<Float[]>(image, RGBColour.WHITE,
					ConnectMode.CONNECT_8);
			br.process(new ConnectedComponent(segment.pixels));
		}
		String this_image = cell_stage + "_all_border_" + image_name;
		ImageUtilities.write(image, new File(path_export, this_image));
	}

	/* ********** BORDER SINGLE MASK USING SHAPE *********** */
	protected void save_cells_single_mask_border(String image_name, MBFImage image,
			List<HoughCircles.WeightedCircle> segments_filtered, String cell_stage, File path_export, boolean shape) // HoughCircles.WeightedCircle
			throws IOException {

		for (final Shape segment : segments_filtered) {
			final MBFImageRenderer renderer = image.createRenderer();
			renderer.drawShape(segment, 2, RGBColour.WHITE);
		}
		
		String this_image = cell_stage + "_all_border_" + image_name;
		ImageUtilities.write(image, new File(path_export, this_image));
	}
	
	/* ********** BORDER SINGLE MASK USING POLY *********** */
	protected void save_cells_single_mask_border_poly(String image_name, MBFImage image,
			List<Polygon> segments_filtered, String cell_stage, File path_export) // HoughCircles.WeightedCircle
			throws IOException {

		for (final Shape segment : segments_filtered) {
			final MBFImageRenderer renderer = image.createRenderer();
			renderer.drawShape(segment, 2, RGBColour.WHITE);
		}
		
		String this_image = cell_stage + "_all_border_" + image_name;
		ImageUtilities.write(image, new File(path_export, this_image));
	}

}
