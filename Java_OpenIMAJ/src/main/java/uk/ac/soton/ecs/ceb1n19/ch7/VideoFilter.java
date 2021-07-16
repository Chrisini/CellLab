package uk.ac.soton.ecs.ceb1n19.ch7;

import java.net.MalformedURLException;
import java.net.URL;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.MBFImage;
// import org.openimaj.image.processing.edges.CannyEdgeDetector;
import org.openimaj.image.processing.edges.SUSANEdgeDetector;
import org.openimaj.video.Video;

import org.openimaj.video.capture.VideoCaptureException;
import org.openimaj.video.xuggle.XuggleVideo;

/**
 * @author Christina Bornberg 31456936
 * 
 * @done true
 * 
 * @task 7.1.1. Exercise 1: Applying different types of image processing to the
 *       video
 *       
 * @resources The OpenIMAJ Tutorial
 */
public class VideoFilter {

	/**
	 * Using different filters like CannyEdge or SusanEdge
	 * 
	 * @throws MalformedURLException
	 * @throws VideoCaptureException
	 */
	private void process() throws MalformedURLException, VideoCaptureException {
		Video<MBFImage> video = new XuggleVideo(new URL("http://static.openimaj.org/media/tutorial/keyboardcat.flv"));
//		video = new XuggleVideo(new File("/path/to/keyboardcat.flv"));
//		video = new VideoCapture(320, 240); // webcam
//		VideoDisplay<MBFImage> display = VideoDisplay.createVideoDisplay(video);
		for (MBFImage mbf_image : video) {
			// DisplayUtilities.displayName(mbf_image.process(new CannyEdgeDetector()), "videoFrames");
			DisplayUtilities.displayName(mbf_image.process(new SUSANEdgeDetector()), "videoFrames");
		}
		video.close();
	}

	public static void main(String[] args) throws MalformedURLException, VideoCaptureException {
		VideoFilter vid = new VideoFilter();
		vid.process();
	}
}
