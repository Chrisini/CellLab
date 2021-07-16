package uk.ac.soton.ecs.ceb1n19.ch6;

import java.io.IOException;
import java.util.Map.Entry;
import java.util.prefs.BackingStoreException;

import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.Image;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.dataset.BingImageDataset;
import org.openimaj.util.api.auth.DefaultTokenFactory;

import org.openimaj.util.api.auth.common.BingAPIToken;
// import org.apache.logging.log4j.Logger;
// import org.apache.logging.log4j.LogManager;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done true
 * 
 * @task 6.1.1. Exercise 1: Exploring Grouped Datasets 6.1.2. Exercise 2: Find
 *       out more about VFS datasets 6.1.3. Exercise 3: Try the BingImageDataset
 *       dataset 6.1.4. Exercise 4: Using MapBackedDataset
 *       
 * @resources The OpenIMAJ Tutorial
 *
 */
public class DataSets {

	/**
	 * Display a random face of each person
	 * 
	 * @throws IOException
	 */
	private void load_face_data() throws IOException {
		VFSGroupDataset<FImage> grouped_faces;

		// Supported File Systems: FTP, HTTP, HTTPS, Jar, Tar, Zip, ...
		grouped_faces = new VFSGroupDataset<FImage>("zip:http://datasets.openimaj.org/att_faces.zip",
				ImageUtilities.FIMAGE_READER);
		for (final Entry<String, VFSListDataset<FImage>> entry : grouped_faces.entrySet()) {
			DisplayUtilities.display(entry.getKey(), entry.getValue());
			DisplayUtilities.display(grouped_faces.getRandomInstance());
		}
	}

	/**
	 * Create key and get 15 images of rockets
	 * 
	 * key: 71a39ee04d6f433c821171eaa845c0f4
	 * 
	 * @throws BackingStoreException
	 */
	private void bing_it() throws BackingStoreException {
		// DefaultTokenFactory.delete(BingAPIToken.class);
		BingAPIToken key = DefaultTokenFactory.get(BingAPIToken.class);
		BingImageDataset<FImage> rocket = BingImageDataset.create(ImageUtilities.FIMAGE_READER, key, "rocket", 15);
		DisplayUtilities.display("Rockets", rocket);
	}

	/**
	 * MapBackedDataset
	 * 
	 * Prints 3 random images from the datasets
	 * 
	 * e.g. 2 fish and 1 hand
	 * 
	 * @throws BackingStoreException
	 */
	private void bing_it_with_map() {
		BingAPIToken key = DefaultTokenFactory.get(BingAPIToken.class);
		BingImageDataset<FImage> rocket = BingImageDataset.create(ImageUtilities.FIMAGE_READER, key, "rocket", 15);
		BingImageDataset<FImage> fish = BingImageDataset.create(ImageUtilities.FIMAGE_READER, key, "fish", 15);
		BingImageDataset<FImage> hand = BingImageDataset.create(ImageUtilities.FIMAGE_READER, key, "hand", 15);
		MapBackedDataset<?, ?, ?> data = MapBackedDataset.of(rocket, fish, hand);
		for (int i = 0; i <= 3; i++) {
			DisplayUtilities.display((Image<?, ?>) data.getRandomInstance(), "A random image from the dataset");
		}
	}

	public static void main(String[] args) throws IOException, BackingStoreException {
		DataSets da = new DataSets();
		da.load_face_data();
		da.bing_it();
		da.bing_it_with_map();
	}

}
