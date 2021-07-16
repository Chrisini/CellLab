package uk.ac.soton.ecs.ceb1;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) {
    	//Create an image
        MBFImage image = new MBFImage(200, 200, ColourSpace.CIE_Lab);

        
        float[][] conv_template = new float[3][3];
        float[][] templ = new float[][]{
        	  { -1, 2, -1},
        	  { 0, 0, 0},
        	  { 1, 2, 1}
        	};
        	
       
        	
        
        int border_row = 8;
		int border_col = 4;
		
		for(int col = 0; col <= border_col; col ++) {
			conv_template[0][col] = (float) 0.5;
			conv_template[border_row][col] = (float) 0.5;
		}
		
		for(int row = 0; row <= border_row; row ++) {
			conv_template[row][0] = (float) 0.5;
			conv_template[row][border_col] = (float) 0.5;
		}
		
		
	    for (int i = 0; i < conv_template.length; i++) {
	        for (int j = 0; j < conv_template[i].length; j++) {
	            System.out.print(conv_template[i][j]);
	            System.out.print(" ");
	        }
	        System.out.println();
	    }
        
        //Fill the image with white
        image.fill(RGBColour.WHITE);
        		        
        //Render some test into the image
        image.drawText("Hello World", 10, 60, HersheyFont.CURSIVE, 50, RGBColour.BLACK);

        //Apply a Gaussian blur
        image.processInplace(new FGaussianConvolve(2f));
        
        //Display the image
        DisplayUtilities.display(image);
        
        /*
         * createFImage(BufferedImage image)
Create an FImage from a buffered image.
         */
    }
}
