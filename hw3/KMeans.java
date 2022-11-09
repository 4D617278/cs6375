/*** Author :Vibhav Gogate
The University of Texas at Dallas
*****/


import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.lang.Math;
import java.util.Random;
 

public class KMeans {
    public static void main(String [] args) {
        int[] test = {2, 5, 10, 15, 20};

        if (args.length < 2){
            // System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
            System.out.println("Usage: Kmeans <input-image> <epochs>");
            return;
        } try {
            File inputFile = new File(args[0]);
            BufferedImage originalImage = ImageIO.read(inputFile);
            long originalSize = inputFile.length();
            int epochs = Integer.parseInt(args[1]);

            for (int i = 0; i < test.length; ++i) {
                double[] ratio = new double[epochs];
                double sum = 0;
                String filename = test[i] + "_" + args[0];

                for (int e = 0; e < ratio.length; ++e) {
                    BufferedImage kmeansJpg = kmeans_helper(originalImage, test[i]);

                    ByteArrayOutputStream tmp = new ByteArrayOutputStream();
                    ImageIO.write(kmeansJpg, "jpg", tmp);
                    tmp.close();

                    ratio[e] = (double)tmp.size() / originalSize;
                    sum += ratio[e];
                        
                    if (e == epochs - 1)
                        ImageIO.write(kmeansJpg, "jpg", new File(filename)); 
                }

                // average
                double avg = sum / ratio.length;

                // sample variance
                double var = 0;
                for (int e = 0; e < ratio.length; ++e)
                    var += Math.pow(ratio[e] - avg, 2);
                var /= (ratio.length - 1);

                String row = args[0] + " & " + test[i] + " & " + avg + " & " + var + " \\\\";
                System.out.println(row);
                System.out.println("\\hline");
            } 

            /*
            int k = Integer.parseInt(args[1]);
            BufferedImage kmeansJpg = kmeans_helper(originalImage, k);
            ImageIO.write(kmeansJpg, "jpg", new File(args[2])); 
            */

        } catch(IOException e) {
            System.out.println(e.getMessage());
        }	
    }
    
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k) {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();
        BufferedImage kmeansImage = new BufferedImage(w, h, originalImage.getType());
        Graphics2D g = kmeansImage.createGraphics();
        g.drawImage(originalImage, 0, 0, w, h, null);

        // Read rgb values from the image
        int[] rgb = new int[w * h];
        int count = 0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                rgb[count++] = kmeansImage.getRGB(i, j);
            }
        }

        // Call kmeans algorithm: update the rgb values
        kmeans(rgb, k);

        // Write the new rgb values to the image
        count = 0;
        for(int i = 0; i < w; i++) {
            for(int j = 0; j < h; j++)
                kmeansImage.setRGB(i, j, rgb[count++]);
        }

        return kmeansImage;
    }

    private static int min_dist_indx(double val, char[] constant) {
        double[] dbl_constant = new double[constant.length];
        for (int i = 0; i < constant.length; ++i)
            dbl_constant[i] = constant[i];

        return min_dist_indx(val, dbl_constant);
    }

    private static int min_dist_indx(double val, double[] constant) {
        double min = Math.pow(val - constant[0], 2);
        int min_i = 0;

        for (int i2 = 1; i2 < constant.length; ++i2) {
            double dist = Math.pow(val - constant[i2], 2);

            if (dist < min) {
                min = dist;
                min_i = i2;
            }
        }

        return min_i;
    }

    // Your k-means code goes here
    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static void kmeans(int[] pixels, int k) {
        if (k <= 0) {
            System.out.println("k > 0");
            return;
        }

        char[][] rgba = new char[Integer.BYTES][pixels.length];

        for (int b = 0; b < rgba.length; ++b) {
            for (int i = 0; i < rgba[b].length; ++i) {
                rgba[b][i] = (char)((pixels[i] >> (Byte.SIZE * b)) & 0xff);
            }
        }

        double[][] means = new double[rgba.length][k];
        int[][] rgba_means = new int[rgba.length][rgba[0].length];
        Random rnd = new Random();

        // kmeans++
        for (int b = 0; b < means.length; ++b)
            means[b][0] = rgba[b][rnd.nextInt(rgba.length)];

        for (int b = 0; b < means.length; ++b) {
            for (int m = 1; m < means[b].length; ++m) {
                double[] min_dist = new double[rgba[b].length];

                // minimum distance to any mean
                for (int i = 0; i < rgba[b].length; ++i) {
                    min_dist[i] = Math.pow(rgba[b][i] - means[b][0], 2);

                    for (int m2 = 1; m2 < m; ++m2) {
                        double dist = Math.pow(rgba[b][i] - means[b][m2], 2);
                        if (dist < min_dist[i])
                            min_dist[i] = dist;
                    }
                }

                double sum = 0;
                for (int i = 0; i < min_dist.length; ++i)
                    sum += min_dist[i];

                double r = rnd.nextDouble() * sum;
                sum = 0;

                for (int i = 0; i < min_dist.length; ++i) {
                    sum += min_dist[i];
                    if (sum >= r) {
                        means[b][m] = rgba[b][i];
                        break;
                    }
                }
            }
        }

        for (int b = 0; b < means.length; ++b) {
            double prev_error, error = Double.MAX_VALUE;

            do {
                    prev_error = error;
                    int[][] num_means = new int[means.length][means[0].length];

                    // assign to mean with minimum distance
                    for (int i = 0; i < rgba[b].length; ++i) {
                        int min_i = min_dist_indx(rgba[b][i], means[b]);
                        rgba_means[b][i] = min_i;
                        num_means[b][min_i] += 1;
                    }

                    // expectation
                    for (int i = 0; i < means[b].length; ++i)
                        means[b][i] = 0;
                    for (int i = 0; i < rgba_means[b].length; ++i)
                        means[b][rgba_means[b][i]] += rgba[b][i];
                    for (int i = 0; i < k; ++i) {
                        if (num_means[b][i] != 0)
                            means[b][i] /= num_means[b][i];
                    }

                    error = 0;
                    for (int i = 0; i < rgba[b].length; ++i)
                        error += Math.pow(means[b][rgba_means[b][i]] - rgba[b][i], 2);
                    // System.out.println("Error: " + error);
            } while (error < prev_error);
        }

        // map mean to closest rgb value
        for (int b = 0; b < rgba.length; ++b) {
            for (int i = 0; i < means[b].length; ++i) {
                int min_i = min_dist_indx(means[b][i], rgba[b]);
                means[b][i] = rgba[b][min_i];
                // System.out.println("b: " + b + " Mean: " + means[b][i]);
            }
        }

        for (int b = 0; b < rgba.length; ++b) {
            for (int i = 0; i < rgba[b].length; ++i) {
                rgba[b][i] = (char)means[b][rgba_means[b][i]];
            }
        }

        for (int i = 0; i < rgba[0].length; ++i) {
            pixels[i] = 0;
            for (int b = 0; b < rgba.length; ++b) {
                pixels[i] += (rgba[b][i] << (Byte.SIZE * b));
            }
        }
    }

}
