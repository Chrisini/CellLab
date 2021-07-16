package uk.ac.soton.ecs.ceb1n19.ch12;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.feature.DiskCachingFeatureExtractor;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
// import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done false, parts are missing
 * 
 * @task 12.1.1. Exercise 1: Apply a Homogeneous Kernel Map, 12.1.2. Exercise 2:
 *       Feature caching 12.1.3. Exercise 3: The whole dataset
 *       
 * @resources The OpenIMAJ Tutorial
 */
class Classifier {

	/**
	 * Classification of images
	 * 
	 * Initial tutorial: Overall Results *********************** Total instances:
	 * 75,000 Total correct: 55,000 Total incorrect: 20,000 Accuracy: 0,733 Error
	 * Rate: 0,267 Average Class Accuracy: 0,733 Average Class Error Rate: 0,267
	 * 
	 * Version 2: Overall Results ****************************** Total instances:
	 * 75,000 Total correct: 60,000 Total incorrect: 15,000 Accuracy: 0,800 Error
	 * Rate: 0,200 Average Class Accuracy: 0,800 Average Class Error Rate: 0,200
	 * 
	 * DiskCachingFeatureExtractor: no result
	 */
	public void classify() throws IOException { // IOException
		// Load data
		GroupedDataset<String, VFSListDataset<Record<FImage>>, Record<FImage>> all_data = Caltech101
				.getData(ImageUtilities.FIMAGE_READER);

		// Create sub-dataset - just 5 groups
		// get rid, if want to get whole dataset
		GroupedDataset<String, ListDataset<Record<FImage>>, Record<FImage>> data = GroupSampler.sample(all_data, 5,
				false);

		// Split to training (15 images per group) and test (15 images per group)
		// dataset
		GroupedRandomSplitter<String, Record<FImage>> splits = new GroupedRandomSplitter<String, Record<FImage>>(data,
				15, 0, 15);

		// Create SIFT objects
		DenseSIFT dsift = new DenseSIFT(3, 7);
		PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 8); // (dsift, 6f, 7);

		// Create hard assigner, random sample of 30 images
		HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(
				GroupedUniformRandomisedSampler.sample(splits.getTrainingDataset(), 30), pdsift);

		// Create feature extractor
		FeatureExtractor<DoubleFV, Record<FImage>> extractor = new PHOWExtractor(pdsift, assigner);
		File cache_dir = new File("C:\\Users\\Prinzessin\\Pictures\\iii.jpg");
		DiskCachingFeatureExtractor<DoubleFV, Record<FImage>> disk_cache = new DiskCachingFeatureExtractor<DoubleFV, Caltech101.Record<FImage>>(
				cache_dir, extractor);

		// Construct and train classifier
		LiblinearAnnotator<Record<FImage>, String> ann = new LiblinearAnnotator<Record<FImage>, String>(disk_cache,
				Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		ann.train(splits.getTrainingDataset());

		ClassificationEvaluator<CMResult<String>, String, Record<FImage>> eval = new ClassificationEvaluator<CMResult<String>, String, Record<FImage>>(
				ann, splits.getTestDataset(), new CMAnalyser<Record<FImage>, String>(CMAnalyser.Strategy.SINGLE));

		// Print results
		Map<Record<FImage>, ClassificationResult<String>> guesses = eval.evaluate();
		CMResult<String> result = eval.analyse(guesses);
		System.out.println("******************");
		System.out.println(result.getDetailReport());

	}

	/**
	 * Hard assigner
	 * 
	 * @param sample
	 * @param pdsift
	 * @return
	 */
	static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<Record<FImage>> sample,
			PyramidDenseSIFT<FImage> pdsift) {
		List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();

		for (Record<FImage> rec : sample) {
			FImage img = rec.getImage();

			pdsift.analyseImage(img);
			allkeys.add(pdsift.getByteKeypoints(0.005f));
		}

		if (allkeys.size() > 10000)
			allkeys = allkeys.subList(0, 10000);

		ByteKMeans km = ByteKMeans.createKDTreeEnsemble(600);
		DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
		ByteCentroidsResult result = km.cluster(datasource);

		return result.defaultHardAssigner();
	}

	/**
	 * Feature extraction
	 * Inner class
	 */
	static class PHOWExtractor implements FeatureExtractor<DoubleFV, Record<FImage>> {

		PyramidDenseSIFT<FImage> pdsift;
		HardAssigner<byte[], float[], IntFloatPair> assigner;

		public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner) {
			this.pdsift = pdsift;
			this.assigner = assigner;
		}

		public DoubleFV extractFeature(Record<FImage> object) {
			FImage image = object.getImage();
			pdsift.analyseImage(image);

			BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

//			BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(bovw,
//					2, 2);
			PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<byte[], SparseIntFV>(
					bovw, 2, 4);

			return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
		}
	}

	public static void main(String[] args) throws IOException {
		Classifier cl = new Classifier();
		cl.classify();
	}
}
