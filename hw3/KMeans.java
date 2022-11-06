/*** Author :Vibhav Gogate
The University of Texas at Dallas
*****/


import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.lang.Math;
import java.util.Random;
 

public class KMeans {
    public static void main(String [] args){
	if (args.length < 3){
	    System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
	    return;
	}
	try{
	    BufferedImage originalImage = ImageIO.read(new File(args[0]));
	    int k=Integer.parseInt(args[1]);
	    BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
	    ImageIO.write(kmeansJpg, "jpg", new File(args[2])); 
	    
	}catch(IOException e){
	    System.out.println(e.getMessage());
	}	
    }
    
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k){
	int w=originalImage.getWidth();
	int h=originalImage.getHeight();
	BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
	Graphics2D g = kmeansImage.createGraphics();
	g.drawImage(originalImage, 0, 0, w,h , null);
	// Read rgb values from the image
	int[] rgb=new int[w*h];
	int count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		rgb[count++]=kmeansImage.getRGB(i,j);
	    }
	}
	// Call kmeans algorithm: update the rgb values
	kmeans(rgb,k);

	// Write the new rgb values to the image
	count=0;
	for(int i=0;i<w;i++){
	    for(int j=0;j<h;j++){
		kmeansImage.setRGB(i,j,rgb[count++]);
	    }
	}
	return kmeansImage;
    }

    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static void kmeans(int[] rgb, int k){
        if (k <= 0) {
            System.out.println("k > 0");
            return;
        }

        if (rgb.length < k) {
            System.out.println("Picture size too small");
            return;
        }

        int[] means = new int[k];
        int[] rgb_means = new int[rgb.length];
        Random rnd = new Random();

        means[0] = rgb[0];
        int uniques = 1;

        int start = rnd.nextInt(rgb.length / k);
        for (int i = start; i < rgb.length && uniques < means.length; ++i) {
            if (rgb[i] > means[uniques - 1]) {
                means[uniques] = rgb[i];
                System.out.println(means[uniques]);
                uniques += 1;
            }
        }

        if (uniques != means.length) {
            System.out.println("Not enough unique pixels");
            return;
        }

        double min_error = 1;
        double error = min_error + 1;

        while (error > min_error) {
            int[] num_means = new int[k];

            // minimization
            for (int i = 0; i < rgb.length; ++i) {
                int min = Math.abs(rgb[i] - means[0]);
                int min_i = 0;

                for (int i2 = 1; i2 < means.length; ++i2) {
                    int diff = Math.abs(rgb[i] - means[i2]);

                    if (diff < min) {
                        min = diff;
                        min_i = i2;
                    }
                }

                rgb_means[i] = min_i;
                num_means[min_i] += 1;
            }

            long[] new_means = new long[k];

            // expectation
            for (int i = 0; i < rgb_means.length; ++i)
                new_means[rgb_means[i]] += rgb[i];
            for (int i = 0; i < new_means.length; ++i)
                if (num_means[i] != 0)
                    new_means[i] /= num_means[i];

            for (int i = 0; i < k; ++i) {
                System.out.println("Mean: " + means[i]);
                System.out.println("New Mean: " + new_means[i]);
            }
                
            error = 0;
            for (int i = 0; i < k; ++i)
                error += Math.abs(new_means[i] - means[i]);
            System.out.println("Error: " + error);

            for (int i = 0; i < k; ++i)
                means[i] = (int)new_means[i];
        }

        // map mean to closest rgb value
        for (int i = 0; i < means.length; ++i) {
            int min = Math.abs(means[i] - rgb[0]);
            int min_i = 0;

            for (int i2 = 1; i2 < rgb.length; ++i2) {
                int diff = Math.abs(means[i] - rgb[i2]);

                if (diff < min) {
                    min = diff;
                    min_i = i2;
                }
            }

            means[i] = rgb[min_i];
        }

        for (int i = 0; i < rgb.length; ++i)
            rgb[i] = means[rgb_means[i]];
    }

}
