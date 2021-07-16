package celllab;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.image.analysis.algorithm.HoughCircles;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.processing.morphology.Close;
import org.openimaj.image.processing.morphology.StructuringElement;

public class PostProcessor {

	protected List<ConnectedComponent> filter_segments_close(List<ConnectedComponent> segments, int min, int max) {

		List<ConnectedComponent> segments_filtered = new ArrayList<>();
		for (ConnectedComponent segment : segments) {
			if (segment.calculateArea() > min && segment.calculateArea() < max) { // 100, 10.000 | 30, 1.000

				Close c = new Close(StructuringElement.CROSS);
				c.process(segment);

				segments_filtered.add(segment);
			}
		}
		return segments_filtered;
	}

	protected List<ConnectedComponent> filter_segments(List<ConnectedComponent> segments, int min, int max) {

		List<ConnectedComponent> segments_filtered = new ArrayList<>();
		for (ConnectedComponent segment : segments) {
			if (segment.calculateArea() > min && segment.calculateArea() < max) {
				segments_filtered.add(segment);
			}
		}
		return segments_filtered;
	}

	protected List<HoughCircles.WeightedCircle> filter_segments_cht(List<HoughCircles.WeightedCircle> segments) {

		List<HoughCircles.WeightedCircle> segments_copy = new ArrayList<HoughCircles.WeightedCircle>(segments);
		
		// remove circles that overlap (minus 3)
		for (int this_cell = 0; this_cell < segments.size(); this_cell++) {
			float this_x = segments.get(this_cell).getX();
			float this_y = segments.get(this_cell).getY();
			float this_radi = segments.get(this_cell).getRadius();

			for (int other_cell = this_cell + 1; other_cell < segments.size(); other_cell++) {
				float other_x = segments.get(other_cell).getX();
				float other_y = segments.get(other_cell).getY();
				float other_radi = segments.get(other_cell).getRadius();

				int x_dist = (int) this_x - (int) other_x;
				int y_dist = (int) this_y - (int) other_y;

				if (Math.sqrt(x_dist * x_dist + y_dist * y_dist) < (this_radi + other_radi - 3)) {
					// remove circle with smaller radius
					if (other_radi < this_radi) {
						segments_copy.remove(segments.get(other_cell));
						System.out.println("here");
					} else {
						segments_copy.remove(segments.get(this_cell));
						System.out.println(segments_copy.size() + " " + segments.size());
					}
				}
			}
		}
		return segments_copy;
	}
}
