package celllab;

import java.util.List;

import org.openimaj.image.Image;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.image.processor.connectedcomponent.render.AbstractRenderer;
import org.openimaj.image.pixel.ConnectedComponent.ConnectMode;

public class CellBorderRenderer<T> extends AbstractRenderer<T> {
	/** The connect mode to use to get the boundary */
	ConnectMode mode;

	/**
	 * Default constructor that takes the image to draw into, the colour to draw the
	 * boundary and the connect mode to use to extract the boundary.
	 * 
	 * @param image  The image to draw into.
	 * @param colour The colour to use to draw the boundary
	 * @param mode   The {@link ConnectMode} to use to extract the boundary.
	 */
	public CellBorderRenderer(Image<T, ?> image, T colour, ConnectMode mode) {
		super(image, colour);
		this.mode = mode;
	}

	/**
	 * Constructor that creates the image to draw into, and takes the colour to draw
	 * the boundary and the connect mode to use to extract the boundary.
	 * 
	 * @param width  The width of the image to create
	 * @param height The height of the image to create
	 * @param colour The colour to use to draw the boundary
	 * @param mode   The {@link ConnectMode} to use to extract the boundary.
	 */
	public CellBorderRenderer(int width, int height, T colour, ConnectMode mode) {
		super(width, height, colour);
		this.mode = mode;
	}

	/**
	 * Draws the boundary of the connected component into the image.
	 * 
	 * {@inheritDoc}
	 * 
	 * @see org.openimaj.image.processor.connectedcomponent.ConnectedComponentProcessor#process(org.openimaj.image.pixel.ConnectedComponent)
	 */
	@Override
	public void process(ConnectedComponent cc) {
		List<Pixel> pset = cc.getInnerBoundary(mode);

		for (Pixel p : pset) {

			image.setPixel(p.x - 1, p.y + 1, colour);
			image.setPixel(p.x - 1, p.y, colour);
			image.setPixel(p.x - 1, p.y - 1, colour);

			image.setPixel(p.x, p.y + 1, colour);
			image.setPixel(p.x, p.y, colour);
			image.setPixel(p.x, p.y - 1, colour);

			image.setPixel(p.x + 1, p.y + 1, colour);
			image.setPixel(p.x + 1, p.y, colour);
			image.setPixel(p.x + 1, p.y - 1, colour);
		}
	}

}
