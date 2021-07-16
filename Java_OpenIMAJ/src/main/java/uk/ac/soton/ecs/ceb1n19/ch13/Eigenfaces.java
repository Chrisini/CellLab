package uk.ac.soton.ecs.ceb1n19.ch13;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.dataset.util.DatasetAdaptors;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.model.EigenImages;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done false, parts are missing
 * 
 * @task 13.1.1. Exercise 1: Reconstructing faces 13.1.2. Exercise 2: Explore
 *       the effect of training set size 13.1.3. Exercise 3: Apply a threshold
 * 
 * @resources The OpenIMAJ Tutorial
 */
public class Eigenfaces {

	/**
	 * Recognition of faces
	 * 
	 * @throws FileSystemException
	 * @throws IOException
	 */
	private void face_rec() throws FileSystemException, IOException {
		/*
		 * By lowering the training set, the accuracy falls Two results per training
		 * data amount, to show, that training varies also within one value
		 * 
		 * Accuracy: 0.935 0.95 (5,5) | Accuracy: 0.915 0.905 (4,5) | Accuracy: 0.845
		 * 0.895 (3,5) | Accuracy: 0.81 0.76 (2,5)
		 */
		int n_training = 4, n_testing = 5;
		int n_eigenvectors = 100;
		double correct = 0, incorrect = 0;

		// Load data set
		VFSGroupDataset<FImage> dataset = new VFSGroupDataset<FImage>("zip:http://datasets.openimaj.org/att_faces.zip",
				ImageUtilities.FIMAGE_READER);

		// Split data set into training / testing data
		GroupedRandomSplitter<String, FImage> splits = new GroupedRandomSplitter<String, FImage>(dataset, n_training, 0,
				n_testing);
		GroupedDataset<String, ListDataset<FImage>, FImage> training = splits.getTrainingDataset();
		GroupedDataset<String, ListDataset<FImage>, FImage> testing = splits.getTestDataset();

		// Training
		List<FImage> basis_images = DatasetAdaptors.asList(training);
		EigenImages eigen = new EigenImages(n_eigenvectors);
		eigen.train(basis_images);

		// Display faces
		List<FImage> eigen_faces = new ArrayList<FImage>();
		for (int i = 0; i < 12; i++) {
			eigen_faces.add(eigen.visualisePC(i));
		}
		DisplayUtilities.display("EigenFaces", eigen_faces);

		// Create feature database
		Map<String, DoubleFV[]> features = new HashMap<String, DoubleFV[]>();
		for (final String person : training.getGroups()) {
			final DoubleFV[] fvs = new DoubleFV[n_training];

			for (int i = 0; i < n_training; i++) {
				final FImage face = training.get(person).get(i);
				fvs[i] = eigen.extractFeature(face);
			}
			features.put(person, fvs);
		}

		// Recognize a person
		for (String true_person : testing.getGroups()) {
			for (FImage face : testing.get(true_person)) {
				DoubleFV test_feature = eigen.extractFeature(face);
				String best_person = null;
				double min_distance = Double.MAX_VALUE;
				for (final String person : features.keySet()) {
					for (final DoubleFV fv : features.get(person)) {
						double distance = fv.compare(test_feature, DoubleFVComparison.EUCLIDEAN);

						if (distance < min_distance) {
							min_distance = distance;
							best_person = person;
						}
					}
				}
				System.out.println("Actual: " + true_person + "\tguess: " + best_person);
				if (true_person.equals(best_person))
					correct++;
				else
					incorrect++;
			}
		}
		System.out.println("Accuracy: " + (correct / (correct + incorrect)));
	}

	public static void main(String[] args) throws FileSystemException, IOException {

		Eigenfaces eg = new Eigenfaces();
		eg.face_rec();

	}

}
