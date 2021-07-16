package uk.ac.soton.ecs.ceb1n19.ch5;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;

//import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.BasicMatcher;
import org.openimaj.feature.local.matcher.BasicTwoWayMatcher;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.FastEuclideanKeypointMatcher;
import org.openimaj.feature.local.matcher.LocalFeatureMatcher;
import org.openimaj.feature.local.matcher.MatchingUtilities;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.math.geometry.point.Point2d;
import org.openimaj.math.geometry.transforms.HomographyModel;
//import org.openimaj.math.geometry.transforms.HomographyRefinement;
//import org.openimaj.math.geometry.transforms.check.TransformMatrixConditionCheck;
import org.openimaj.math.geometry.transforms.estimation.RobustAffineTransformEstimator;
//import org.openimaj.math.geometry.transforms.estimation.RobustHomographyEstimator;
import org.openimaj.math.geometry.transforms.residuals.SingleImageTransferResidual2d;
import org.openimaj.math.model.fit.RANSAC;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done true
 * 
 * @task 5.1.1. Exercise 1: Different matchers 5.1.2. Exercise 2: Different models
 * 
 * @resources The OpenIMAJ Tutorial
 */
public class Matcher {

	/**
	 * Source:
	 * https://github.com/openimaj/openimaj/blob/master/image/image-local-features/src/main/java/org/openimaj/feature/local/matcher/FastEuclideanKeypointMatcher.java
	 * 
	 * This matcher cannot deal with rotated images, plus changing the threshold has
	 * no effect on the matches
	 * 
	 * @throws MalformedURLException
	 * @throws IOException
	 */
	private void match_euclidean(int thresh) throws MalformedURLException, IOException {
		MBFImage query = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/query.jpg"));
		MBFImage target = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/query.jpg"));

		// SIFT
		DoGSIFTEngine engine = new DoGSIFTEngine();
		LocalFeatureList<Keypoint> query_keypoints = engine.findFeatures(query.flatten());
		LocalFeatureList<Keypoint> target_keypoints = engine.findFeatures(target.flatten());

		// Fast Eucledian Keypoint Matcher
		FastEuclideanKeypointMatcher<Keypoint> matcher = new FastEuclideanKeypointMatcher<Keypoint>(thresh);
		matcher.setModelFeatures(query_keypoints);
		matcher.findMatches(target_keypoints);

		MBFImage draw_matches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
		DisplayUtilities.display(draw_matches, "Euclidean");
	}

	/**
	 * Two Way Matcher had quite bad results, could be improved by filtering the matches
	 * 
	 * Source:
	 * https://github.com/openimaj/openimaj/blob/master/image/image-local-features/src/main/java/org/openimaj/feature/local/matcher/BasicTwoWayMatcher.java
	 * 
	 * @throws MalformedURLException
	 * @throws IOException
	 */
	private void match_twoway() throws MalformedURLException, IOException {
		MBFImage query = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/query.jpg"));
		MBFImage target = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/target.jpg"));

		// SIFT
		DoGSIFTEngine engine = new DoGSIFTEngine();
		LocalFeatureList<Keypoint> query_keypoints = engine.findFeatures(query.flatten());
		LocalFeatureList<Keypoint> target_keypoints = engine.findFeatures(target.flatten());

		// Basic Two Way Matcher
		BasicTwoWayMatcher<Keypoint> matcher = new BasicTwoWayMatcher<Keypoint>();
		matcher.setModelFeatures(query_keypoints);
		matcher.findMatches(target_keypoints);
		MBFImage consistent_matches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
		DisplayUtilities.display(consistent_matches, "Basic two way matcher");
	}
	
	/**
	 * Homography Model shows quite a good result
	 * 
	 * Source:
	 * https://www.programcreek.com/java-api-examples/?api=org.openimaj.math.geometry.transforms.HomographyModel
	 * 
	 * @throws IOException
	 * @throws MalformedURLException
	 */
	private void homography_model() throws MalformedURLException, IOException {
		MBFImage query = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/query.jpg"));
		MBFImage target = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/target.jpg"));

		// SIFT
		DoGSIFTEngine engine = new DoGSIFTEngine();
		LocalFeatureList<Keypoint> query_keypoints = engine.findFeatures(query.flatten());
		LocalFeatureList<Keypoint> target_keypoints = engine.findFeatures(target.flatten());
		
		// Homography Model
		final HomographyModel model_fitter = new HomographyModel();
		final SingleImageTransferResidual2d<HomographyModel> errorModel = new SingleImageTransferResidual2d<HomographyModel>();
		final RANSAC<Point2d, Point2d, HomographyModel> ransac = new RANSAC<Point2d, Point2d, HomographyModel>(model_fitter,
				errorModel, 5f, 1500, new RANSAC.BestFitStoppingCondition(), true);
		final ConsistentLocalFeatureMatcher2d<Keypoint> matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(
				new FastBasicKeypointMatcher<Keypoint>(8));
		matcher.setFittingModel(ransac);

		// Set features and find matches
		matcher.setModelFeatures(query_keypoints);
		matcher.findMatches(target_keypoints);

		// Display image with matches
		MBFImage consistent_matches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
		DisplayUtilities.display(consistent_matches, "Homography Model");
	}

	/**
	 * LocalFeatureMatcher and ConsistentLocalFeatureMatcher2d were used
	 * to match points in a picture
	 * Image can rotate here, it still works
	 * Better results when filtering the points with RobustAffineTransformEstimator and RANSAC
	 * 
	 * @throws MalformedURLException
	 * @throws IOException
	 */
	private void match_local_feature() throws MalformedURLException, IOException {
		MBFImage query = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/query.jpg"));
		MBFImage target = ImageUtilities.readMBF(new URL("http://static.openimaj.org/media/tutorial/target.jpg"));

		// SIFT
		DoGSIFTEngine engine = new DoGSIFTEngine();
		LocalFeatureList<Keypoint> query_keypoints = engine.findFeatures(query.flatten());
		LocalFeatureList<Keypoint> target_keypoints = engine.findFeatures(target.flatten());

		// Basic Matcher - not very effective
		LocalFeatureMatcher<Keypoint> matcher = new BasicMatcher<Keypoint>(80);

		// Filter the matches
		RobustAffineTransformEstimator model_fitter = new RobustAffineTransformEstimator(5.0, 1500,
				new RANSAC.PercentageInliersStoppingCondition(0.5));
		matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(new FastBasicKeypointMatcher<Keypoint>(8),
				model_fitter);

		// Set features and find matches
		matcher.setModelFeatures(query_keypoints);
		matcher.findMatches(target_keypoints);

		// Display image with matches
		MBFImage consistent_matches = MatchingUtilities.drawMatches(query, target, matcher.getMatches(), RGBColour.RED);
		DisplayUtilities.display(consistent_matches, "Local Feature Matcher");

//		target.drawShape(
//				query.getBounds().transform(modelFitter.getModel().getTransform().inverse()), 3, RGBColour.BLUE);
//		DisplayUtilities.display(target);

	}

	public static void main(String[] args) throws MalformedURLException, IOException {
		int thresh = 70;
		Matcher m = new Matcher();
		m.match_local_feature();
		m.match_euclidean(thresh);
		m.match_twoway();
		m.homography_model();

	}

}
