package uk.ac.soton.ecs.ceb1n19.ch14;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.time.Timer;
import org.openimaj.util.function.Operation;
import org.openimaj.util.parallel.Parallel;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done true
 * 
 * @task 14.1.1. Exercise 1: Parallelize the outer loop
 * 
 * @resources The OpenIMAJ Tutorial
 */
public class ParallelProc {

	/**
	 * Instead of using a loop, the Parallel class can be used
	 * 
	 * With inner and outer loop: 24526ms
	 * 
	 * With outer loop and inner parallel: 9541ms
	 * 
	 * With inner and outer parallel: - 8082ms without sync function
	 * 
	 * With outer parallel and inner loop: - 16025ms without sync - 47962ms with
	 * sync (current) - 18588ms with sync (output) The synchronisation of the
	 * different values seems to be time intensive
	 * 
	 * Last result: Loops: Time: 35311ms - slowest Inner Parallel: Time: 8350ms -
	 * fastest Outer Parallel: Time: 15590ms - middle
	 * 
	 * @throws IOException
	 */
	private void compare_time() throws IOException {

		// Load dataset and create sub-dataset with 8 groups
		VFSGroupDataset<MBFImage> allImages = Caltech101.getImages(ImageUtilities.MBFIMAGE_READER);
		GroupedDataset<String, ListDataset<MBFImage>, MBFImage> images = GroupSampler.sample(allImages, 8, false);

		// Compare
		System.out.print("Loops: ");
		using_loops(images);
		System.out.print("Inner Parallel: ");
		using_inner_parallel(images);
		System.out.print("Outer Parallel: ");
		using_outer_parallel(images);
	}

	private void using_loops(GroupedDataset<String, ListDataset<MBFImage>, MBFImage> images) {
		List<MBFImage> output = new ArrayList<MBFImage>();
		ResizeProcessor resize = new ResizeProcessor(200);
		Timer t1 = Timer.timer(); // Set timer
		// Outer loop
		for (ListDataset<MBFImage> clz_images : images.values()) {
			MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);
			// Inner loop
			for (MBFImage i : clz_images) {
				MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
				tmp.fill(RGBColour.WHITE);
				MBFImage small = i.process(resize).normalise();
				int x = (200 - small.getWidth()) / 2;
				int y = (200 - small.getHeight()) / 2;
				tmp.drawImage(small, x, y);
				current.addInplace(tmp);
			}
			current.divideInplace((float) clz_images.size());
			output.add(current);
		}
		System.out.println("Time: " + t1.duration() + "ms"); // Read timer
		DisplayUtilities.display("Images", output);
	}

	private void using_inner_parallel(GroupedDataset<String, ListDataset<MBFImage>, MBFImage> images) {
		List<MBFImage> output = new ArrayList<MBFImage>();
		final ResizeProcessor resize = new ResizeProcessor(200);
		Timer t1 = Timer.timer();
		// Outer loop
		for (ListDataset<MBFImage> clz_images : images.values()) {
			final MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);
			// Inner loop
			Parallel.forEach(clz_images, new Operation<MBFImage>() {
				public void perform(MBFImage i) {
					final MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
					tmp.fill(RGBColour.WHITE);
					final MBFImage small = i.process(resize).normalise();
					final int x = (200 - small.getWidth()) / 2;
					final int y = (200 - small.getHeight()) / 2;
					tmp.drawImage(small, x, y);
					synchronized (current) {
						current.addInplace(tmp);
					}
				}
			});
			current.divideInplace((float) clz_images.size());
			output.add(current);
		}
		System.out.println("Time: " + t1.duration() + "ms");
		DisplayUtilities.display("Images", output);
	}

	private void using_outer_parallel(GroupedDataset<String, ListDataset<MBFImage>, MBFImage> images) {
		final List<MBFImage> output = new ArrayList<MBFImage>();
		final ResizeProcessor resize = new ResizeProcessor(200);
		Timer t1 = Timer.timer();

		// Outer loop
		Parallel.forEach(images.values(), new Operation<ListDataset<MBFImage>>() {
			public void perform(ListDataset<MBFImage> clz_images) {
				final MBFImage current = new MBFImage(200, 200, ColourSpace.RGB);
				// Inner loop
				for (MBFImage i : clz_images) {
					MBFImage tmp = new MBFImage(200, 200, ColourSpace.RGB);
					tmp.fill(RGBColour.WHITE);
					MBFImage small = i.process(resize).normalise();
					int x = (200 - small.getWidth()) / 2;
					int y = (200 - small.getHeight()) / 2;
					tmp.drawImage(small, x, y);
					current.addInplace(tmp);
				}
				current.divideInplace((float) clz_images.size());
				synchronized (output) {
					output.add(current);
				}
			}
		});

		System.out.println("Time: " + t1.duration() + "ms");
		DisplayUtilities.display("Images", output);
	}

	private void test_parallel() {
		// Numbers are printed, but in wrong order: (e.g.) 0, 1, 5, 4, 2, ...
		Parallel.forIndex(0, 10, 1, new Operation<Integer>() {
			public void perform(Integer i) {
				System.out.print(i + " ");
			}
		});
		System.out.println();
	}

	public static void main(String[] args) throws IOException {
		ParallelProc par = new ParallelProc();
		par.test_parallel();
		par.compare_time();
		System.out.println("End");
	}

}
