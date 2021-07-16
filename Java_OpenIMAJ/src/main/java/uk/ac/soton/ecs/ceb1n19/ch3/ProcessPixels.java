package uk.ac.soton.ecs.ceb1n19.ch3;

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
import org.openimaj.image.pixel.PixelSet;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.convolution.FSobel;
import org.openimaj.image.processing.edges.StrokeWidthTransform;
import org.openimaj.image.processor.PixelProcessor;
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

/**
 * @author Christina Bornberg 31456936
 * 
 * @done true
 * 
 * @task 3.1.1. Exercise 1: The PixelProcessor 3.1.2. Exercise 2: A real
 *       segmentation algorithm
 * 
 * @resources The OpenIMAJ Tutorial
 */
public class ProcessPixels {
	
	
	private void hough_algorithm() throws IOException {
		// actually not hough, but only feature detector
		// not working ..... but it's the code of twitter
		
		FImage image2 = ImageUtilities.readF(new File("C:\\Users\\Prinzessin\\compvis/cell49.jpg")); //something.jpg")); //
		MBFImage image = ImageUtilities.readMBF(new File("C:\\\\Users\\\\Prinzessin\\\\compvis/cell49.jpg"));
		MBFImage blur = ImageUtilities.readMBF(new File("C:\\\\Users\\\\Prinzessin\\\\compvis/cell49.jpg"));
	
		final MBFImageRenderer renderer = image.createRenderer();
		
	
		blur.processInplace(new FGaussianConvolve(100f));

		// Jon: here I subtract my image from the blur image.
		//image = image.subtract(blur);
		
		
		
/*		
		HoughCircles hc = new HoughCircles(1, 100, 1, 1);
		hc.analyseImage(image2);
		List<HoughCircles.WeightedCircle> listhc = hc.getBest(10);
		
		if (listhc.size() != 0){
			System.out.println(listhc);
			System.out.println(listhc.size());
		}else {
			System.out.println("no cirlces found");
			return;
		}
			
		
		final List<Circle> circs = new ArrayList<Circle>();
//		circs.add(new Circle(listhc.get(0).getX(), listhc.get(0).getY(), listhc.get(0).getRadius()));
		
		System.out.println(circs);
		
		  for (final Circle cir : circs) {
		        renderer.drawShapeFilled(cir, RGBColour.BLACK);
		    }
		*/
		
		float[][] s = new float[500][3];
		int i = 0;
		  final MSERFeatureGenerator mser = new MSERFeatureGenerator(MomentFeature.class);
		  //final MSERFeatureGenerator mser = new MSERFeatureGenerator(1, 1000, 1, 0.4f, 1, MomentFeature.class);
		  
		  //int delta, int maxArea, int minArea, float maxVariation, float minDiversity, Class<? extends ComponentFeature>... featureClasses
		  
		    final List<Component> features = mser.generateMSERs(Transforms.calculateIntensityNTSC(image));
		    final List<Component> features2 = new ArrayList<>();
		    
		    
		    // jedes feature, das eine bestimmte größe hat objA
		    for (final Component c : features) {
		        final MomentFeature oA = c.getFeature(MomentFeature.class);
		        final MomentFeature oC = c.getFeature(MomentFeature.class);
		        
		        if (oA.n() < 300 && oA.n() > 250) { // amount of accumulator pixels
		        	
		        	// gehe durch die liste objB
		        	for(final Component cc : features) {
		        		 final MomentFeature oB = cc.getFeature(MomentFeature.class);
		        		 if (oB.n() < 300 && oB.n() > 250) {
		        			 System.out.println(oB.n());
		        			 
		        		 // wenn oA und oB in der nähe liegen
		        		 if(oB.getCircle(2).getX() < oA.getCircle(2).getX()+20 &&
		        				 oB.getCircle(2).getX() > oA.getCircle(2).getX()-20) {
		        			 if(oB.getCircle(2).getY() < oA.getCircle(2).getY()+20 &&
		        					 oB.getCircle(2).getY() > oA.getCircle(2).getY()-20) {
		        				 
		        				 // speichere ein +1 in die liste
		        				 
		        				 
					        	s [i][2] = s[i][2]+1;
					        	//System.out.println(s[i][2]);
					        	
					        	if(s [i][2] == 2.0) {
					        		 renderer.drawShape(oB.getCircle(2), RGBColour.RED);
					        		 renderer.drawShape(oB.getCircle(i), RGBColour.RED);
					        	}
					        	
		        			 }else {
		        				 // if new
		        				 s [i][0] = oB.getCircle(2).getX();
		        				 s [i][1] = oB.getCircle(2).getY();
		        				 s [i][2] = 1; 
		        				 renderer.drawShape(oB.getCircle(i), RGBColour.RED);
		        				 if (i>(500-10)) {
		        					 break;
		        				 }
		        						i++;
		        				 System.out.println("New object");
		        			 }
		        		
		        		 }
		        		 }
			        	}
		        	
		        	
		        	

		        	
				  //      System.out.println(feature.getCircle(2).getX() + " " + feature.getCircle(2).getY());
				        
				       
//				        renderer.drawShape(feature.getEllipse(2)
//				                .calculateOrientedBoundingBox(), RGBColour.GREEN);
		        }
		       
		    }
		
		DisplayUtilities.display(image);
		
		
	}
	
	private void edge_algorithm() throws IOException{
		
		FImage image = ImageUtilities.readF(new File("D:\\no/alive_multi.jpg"));
		
		float sigma = 10f;
		
		FSobel fs = new FSobel();
		
		fs.analyseImage(image);
		
		//processImage(image, new FSobel(sigma));
		  
	}

	/**
	 * Using a pixel processor
	 * 
	 * @throws MalformedURLException
	 * @throws IOException
	 */
	private void pixel_processor() throws MalformedURLException, IOException {

		// new
		// VFSListDataset<MBFImage>("C:\\Users/Prinzessin/Documents/LifeSci/celllab_train/all",
		// ImageUtilities.MBFIMAGE_READER);

		// Loading image vie URL
		MBFImage image = new MBFImage();
		image = ImageUtilities.readMBF(new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\sicherung_don't change\\chicken_multi.jpg"));
		MBFImage image2 = new MBFImage();
		image2 = ImageUtilities.readMBF(new File("C:\\\\Users\\\\Prinzessin\\\\Documents\\\\LifeSci\\\\sicherung_don't change\\\\chicken_multi.jpg"));
		MBFImage imageblur = new MBFImage();
		MBFImage rgb_display = new MBFImage();
		imageblur = ImageUtilities.readMBF(new File("C:\\\\Users\\\\Prinzessin\\\\Documents\\\\LifeSci\\\\sicherung_don't change\\\\chicken_multi.jpg"));
		imageblur.processInplace(new FGaussianConvolve(100f));

		// MBFImage image = new MBFImage();

//		for (MBFImage image: images) {
//			MBFImage image2;
//			MBFImage imageblur;
//			imageblur.processInplace(new FGaussianConvolve(100f));

		// Jon: here I subtract my image from the blur image.
		//image = image.subtract(imageblur);
		
		// ImageUtilities.write(image, new File("D:\\new\\subtraction.jpg"));

		/*
		 * for (int y=0; y<image.getHeight(); y++) { for(int x=0; x<image.getWidth();
		 * x++) { System.out.print(image.getBand(1).pixels[y][x]); } }
		 */

		DisplayUtilities.display(image, "Original");

		// Colour space transformation
		image = ColourSpace.convert(image, ColourSpace.CIE_Lab);
		imageblur = ColourSpace.convert(image, ColourSpace.CIE_Lab);
		DisplayUtilities.display(image, "CIELab image");

		// Construct K-Means algorithm
		FloatKMeans cluster = FloatKMeans.createExact(4); // 3 or >4< Colours!!! 5 works better for sticky cells

		// Flatten and group pixels
		float[][] image_data = image.getPixelVectorNative(new float[image.getWidth() * image.getHeight()][3]);
		FloatCentroidsResult result = cluster.cluster(image_data);

		// rgb_display = ColourSpace.convert(image, ColourSpace.RGB);
		// DisplayUtilities.display(rgb_display);

		// Jon: Here, I get a negative value for L, for my run it was:
		// [-8.1692705, 1.6392353, -2.395253]
		// Print centroid: (L, a, b) - from CIE_Lab colourspace
		final float[][] centroids = result.centroids;
		float thresh = (float) -100.0;
		for (float[] fs : centroids) {
			System.out.println(Arrays.toString(fs));
			if (thresh < fs[1]) {
				thresh = fs[1];
			}
		}

		final float thr = thresh;
		System.out.println(thr);
		// Hard assigner for the image
		final HardAssigner<float[], ?, ?> assigner = result.defaultHardAssigner();

		/*
		 * The Pixel processor automatically loops through the image. By using
		 * System.out.println(Arrays.toString(pixel)); each pixel with the dimensions L,
		 * a, b are printed Example for one line: [41.899956, 4.2016478, 9.808258]
		 * 
		 * The pixel processor addresses each pixel, the developer doesn't need to care
		 * about the position of the pixel. The only problem is the cast. Arrays can't
		 * be copied/accessed without a loop.
		 */
		image.processInplace(new PixelProcessor<Float[]>() {
			public Float[] processPixel(Float[] pixel) {
				float[] fp = new float[3];
				for (int lab = 0; lab < 3; lab++) {
					fp[lab] = pixel[lab];
				}
				final int centroid = assigner.assign(fp);
				System.out.println(centroid);

				// System.out.println(Arrays.toString(centroids[centroid]));
				// for (int lab = 0; lab < 3; lab++) {

				if (centroids[centroid][1] >= thr) {// > 30.0 && centroids[centroid][lab] < 50.0) {
					pixel[0] = (float) 0.0;
					pixel[1] = (float) 0.0;
					pixel[2] = (float) 0.0;
				} else {
					pixel[0] = (float) 50.1;
					pixel[1] = (float) 0.1;
					pixel[2] = (float) -20.1;
				}

				/*
				 * pixel[0] = centroids[centroid][0]; pixel[1] = centroids[centroid][1];
				 * pixel[2] = centroids[centroid][2];
				 */

				return pixel;
			}
		});

		imageblur.processInplace(new PixelProcessor<Float[]>() {
			public Float[] processPixel(Float[] pixel) {
				float[] fp = new float[3];
				for (int lab = 0; lab < 3; lab++) {
					fp[lab] = pixel[lab];
				}
				final int centroid = assigner.assign(fp);

				// System.out.println(Arrays.toString(centroids[centroid]));
				for (int lab = 0; lab < 3; lab++) {
					pixel[lab] = centroids[centroid][lab];
				}

				/*
				 * pixel[0] = centroids[centroid][0]; pixel[1] = centroids[centroid][1];
				 * pixel[2] = centroids[centroid][2];
				 */

				return pixel;
			}
		});

		imageblur = ColourSpace.convert(imageblur, ColourSpace.RGB);
		// ImageUtilities.write(imageblur, new
		// File("D:\\new\\inhib_multi_kmeans_a4_new.jpg"));

		image = ColourSpace.convert(image, ColourSpace.RGB);
		// ImageUtilities.write(image, new File("D:\\new\\alive_single_kmeans_b4.jpg"));

		DisplayUtilities.display(imageblur, "4 classes");

		GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
		List<ConnectedComponent> components = labeler.findComponents(image.flatten());
		List<ConnectedComponent> components2 = new ArrayList<>();

		for (ConnectedComponent comp : components) {
			if (comp.calculateArea() > 30 && comp.calculateArea() < 1000) {
				components2.add(comp);
			}
		}

		OrientatedBoundingBoxRenderer<Float[]> obbr = new OrientatedBoundingBoxRenderer<Float[]>(image2, RGBColour.BLUE);

		ConnectedComponent.process(components2, obbr);

		// ImageUtilities.write(image2, new File("D:\\kmeans_LABELS_blue.jpg"));
		DisplayUtilities.display(image2, "boxes");

	}

	/**
	 * Felzenszwalb Huttenlocher Segmenter
	 * 
	 * Used website:
	 * https://www.programcreek.com/java-api-examples/?class=org.openimaj.image.segmentation.FelzenszwalbHuttenlocherSegmenter&method=segment
	 * 
	 * With the segmenter, multiple objects were segmented. For binary
	 * classification, a HardAssigner or similar would need to be used again.
	 * 
	 * @throws IOException
	 * @throws MalformedURLException
	 */
	private void seg_algorithm() throws MalformedURLException, IOException {

		MBFImage image = new MBFImage();
		image = ImageUtilities.readMBF(new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\sicherung_don't change\\chicken_multi.jpg"));
		MBFImage original = new MBFImage();
		original = ImageUtilities.readMBF(new File("C:\\Users\\Prinzessin\\Documents\\LifeSci\\sicherung_don't change\\chicken_multi.jpg"));

		// create new segmenter
		// 	protected float sigma = 0.5f;
		//protected float k = 500f / 255f;
		//protected int minSize = 50;
		FelzenszwalbHuttenlocherSegmenter<MBFImage> fhs = new FelzenszwalbHuttenlocherSegmenter<MBFImage>(0.5f, 500f/200f, 50);

		// Put segments into a list
		final List<? extends PixelSet> segments = fhs.segment(image);

		// Render segments
		image = SegmentationUtilities.renderSegments(image, segments); //image.getWidth(), image.getHeight()
		
		GreyscaleConnectedComponentLabeler labeler = new GreyscaleConnectedComponentLabeler();
		List<ConnectedComponent> components = labeler.findComponents(image.flatten());
		List<ConnectedComponent> components2 = new ArrayList<>();

		for (ConnectedComponent comp : components) {
			if (comp.calculateArea() > 5 && comp.calculateArea() < 10000) {
				components2.add(comp);
			}
		}

		OrientatedBoundingBoxRenderer<Float[]> obbr = new OrientatedBoundingBoxRenderer<Float[]>(original, RGBColour.BLUE);

		ConnectedComponent.process(components2, obbr);
		

		// Export image as jpg file
//		final File path = new File("C:\\Users\\Prinzessin\\Pictures");
//		ImageUtilities.write(image, new File(path, "chrisy8.jpg"));

		//ImageUtilities.write(original, new File("D:\\felz_dead_blue.jpg"));
		DisplayUtilities.display(image);
		DisplayUtilities.display(original);
	}

/*	void histogram_analysis()  throws MalformedURLException, IOException {
		FImage img = new FImage( ... );
 		HistogramProcessor hp = new HistogramProcessor( 64 );
 		img.analyse( hp );
 		Histogram h = hp.getHistogram();
	}
*/	
	public static void main(String[] args) throws MalformedURLException, IOException {
		
//		ProcessPixels pix = new ProcessPixels();
//		pix.pixel_processor();

		 ProcessPixels seg = new ProcessPixels();
		 seg.seg_algorithm();
		 
//		 ProcessPixels cht = new ProcessPixels();
//		 cht.hough_algorithm();
	}
}