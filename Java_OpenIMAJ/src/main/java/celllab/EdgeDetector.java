package celllab;

import java.io.File;
import java.io.IOException;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.convolution.FSobelX;
import org.openimaj.image.processing.convolution.FSobelY;
import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.processing.edges.NonMaximumSuppressionTangent;

import com.jogamp.newt.Display;

/*
 * I used an unofficial edge detector: https://github.com/wuchubuzai/OpenIMAJ/blob/master/image/image-processing/src/main/java/org/openimaj/image/processing/edges/CannyEdgeDetector.java
 */

public class EdgeDetector {
	
	float threshold = 25f / 255f;
	float hyst_threshold_1 = 30f / 255f;
	float hyst_threshold_2 = 230f / 255f;
	float sigma = 1;
	
	private void thresholding_tracker(float threshMin, float threshMax, FImage magnitude, FImage output) {
		output.zero();
 
		for (int i1 = 0; i1 < magnitude.height; i1++) {
			for (int l = 0; l < magnitude.width; l++) {
				if (magnitude.pixels[i1][l] >= threshMin) {
					follow(l, i1, threshMax, magnitude, output);
				}
			}
		}
	}
 
	private boolean follow(int i, int j, float threshMax, FImage magnitude, FImage output) {
		int j1 = i + 1;
		int k1 = i - 1;
		int l1 = j + 1;
		int i2 = j - 1;
		
		if (l1 >= magnitude.height) l1 = magnitude.height - 1;
		if (i2 < 0) i2 = 0;
		if (j1 >= magnitude.width) j1 = magnitude.width - 1;
		if (k1 < 0) k1 = 0;
		
		if (output.pixels[j][i] == 0) {
			output.pixels[j][i] = magnitude.pixels[j][i];
			boolean flag = false;
			int l = k1;
			do {
				if (l > j1) break;
				int i1 = i2;
				do {
					if (i1 > l1) break;
					
					if ((i1 != j || l != i)
						&& magnitude.pixels[i1][l] >= threshMax
						&& follow(l, i1, threshMax, magnitude, output)) {
						flag = true;
						break;
					}
					i1++;
				} while (true);
				if (!flag)
					break;
				l++;
			}
			while (true);
			return true;
		} else {
			return false;
		}
	}
	
	public void processImage(FImage image) {
		FImage tmp = image.process(new FGaussianConvolve(sigma));		
		
		FImage dx = tmp.process(new FSobelX());
		FImage dy = tmp.process(new FSobelY());
		
		FImage magnitudes = NonMaximumSuppressionTangent.computeSuppressed(dx, dy);
		thresholding_tracker(hyst_threshold_1, hyst_threshold_2, magnitudes, image);
		image.threshold(threshold);
	}

	

	public static void main(String[] args) throws IOException {
		
		
		String path = "C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_images_type\\dead\\cell60.jpg";
		FImage image = ImageUtilities.readF(new File(path));
		FImage image2 = image.clone();
		
		EdgeDetector detector = new EdgeDetector();
		detector.processImage(image);
		
		CannyEdgeDetector canny = new CannyEdgeDetector((float) 0.01, (float) 0.2, 1);
		canny.processImage(image2);
		
		DisplayUtilities.display(image, "edge");
		DisplayUtilities.display(image2, "canny");

	}

}
