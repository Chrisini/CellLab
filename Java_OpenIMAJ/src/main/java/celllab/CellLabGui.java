package celllab;


import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

public class CellLabGui extends JFrame{

    
    public CellLabGui(String name) {
        super(name);
    }
    
    public void addComponentsToPane(final Container pane) {
        
        JPanel image_panel = new JPanel();
        image_panel.setLayout(new GridLayout(1,2));      
        
        ImageIcon img_icon = new ImageIcon("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\pipeline_matlab_morph\\inhib_all_cell115.jpg");
        
        Image image_im = img_icon.getImage(); // transform it 
        Image newimg = image_im.getScaledInstance(500, 500,  java.awt.Image.SCALE_SMOOTH); // scale it the smooth way  
        img_icon = new ImageIcon(newimg);  // transform it back
        image_panel.add(new JLabel(img_icon));
        
        ImageIcon mask_icon = new ImageIcon("C:\\Users\\Prinzessin\\Documents\\LifeSci\\named_comp_masks\\pipeline_matlab_morph\\inhib_all_cell115.jpg");
        Image mask_im = mask_icon.getImage(); // transform it 
        Image newmask = mask_im.getScaledInstance(500, 500,  java.awt.Image.SCALE_SMOOTH); // scale it the smooth way  
        mask_icon = new ImageIcon(newmask);  // transform it back
        image_panel.add(new JLabel(mask_icon));
        
        
        // CONTROLS
        
        JPanel controls_panel = new JPanel(new BorderLayout());
        
        JPanel controls_panel_north = new JPanel();
        controls_panel_north.setLayout(new GridLayout(1,2));
                
        //switch image
        controls_panel_north.add(new JButton("Previous"));
        controls_panel_north.add(new JButton("Next"));
        controls_panel.add(controls_panel_north, BorderLayout.NORTH);
        
        JPanel controls_panel_center = new JPanel();
        controls_panel_center.setLayout(new GridLayout(1,6));
        
        // algorithms
        controls_panel_center.add(new Label("Choose algorithm:"));
        controls_panel_center.add(new JButton("CHT"));
        controls_panel_center.add(new JButton("k-means"));
        controls_panel_center.add(new JButton("Graph"));
        controls_panel_center.add(new JButton("Edge"));
        controls_panel_center.add(new JButton("Threshold"));
        controls_panel.add(controls_panel_center, BorderLayout.CENTER);
        
        
        JPanel controls_panel_south = new JPanel();
        controls_panel_south.setLayout(new GridLayout(3,3));

        // preprocessing, settings and filtering size
        controls_panel_south.add(new JButton("apply"));
        controls_panel_south.add(new JButton("apply"));
        controls_panel_south.add(new JButton("apply"));
        controls_panel.add(controls_panel_south, BorderLayout.SOUTH);

        JPanel save_panel = new JPanel();
        save_panel.setLayout(new GridLayout(1,6));
        //cell stage
        save_panel.add(new Label("Choose cell type and save:"));
        save_panel.add(new JButton("Mostly Alive"));
        save_panel.add(new JButton("Mostly Inhib"));
        save_panel.add(new JButton("Mostly Dead"));
        save_panel.add(new JButton("Contains Fibre"));
        

        pane.add(image_panel, BorderLayout.NORTH);
        
        pane.add(controls_panel, BorderLayout.CENTER);
        
        pane.add(save_panel, BorderLayout.SOUTH);
    }
    
    /**
     * Create the GUI and show it.  For thread safety,
     * this method is invoked from the
     * event dispatch thread.
     */
    private static void createAndShowGUI() {
        //Create and set up the window.
    	CellLabGui frame = new CellLabGui("CellLab GUI");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        //Set up the content pane.
        frame.addComponentsToPane(frame.getContentPane());
        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }
    
    public static void main(String[] args) {
        /* Use an appropriate Look and Feel */
        try {
            //UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
            UIManager.setLookAndFeel("javax.swing.plaf.metal.MetalLookAndFeel");
        } catch (UnsupportedLookAndFeelException ex) {
            ex.printStackTrace();
        } catch (IllegalAccessException ex) {
            ex.printStackTrace();
        } catch (InstantiationException ex) {
            ex.printStackTrace();
        } catch (ClassNotFoundException ex) {
            ex.printStackTrace();
        }
        /* Turn off metal's use of bold fonts */
        UIManager.put("swing.boldMetal", Boolean.FALSE);
        
        //Schedule a job for the event dispatch thread:
        //creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                createAndShowGUI();
            }
        });
    }
}
