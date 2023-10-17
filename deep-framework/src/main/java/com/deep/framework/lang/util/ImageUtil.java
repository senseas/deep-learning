package com.deep.framework.lang.util;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

/**
 * @description:
 * @author: cdl
 * @create: 2021-03-25 17:54
 **/
public class ImageUtil {

    public static double[][][] image2RGB(String src) {
        try {
            InputStream input = FileUtil.readResourceAsStream(src);
            BufferedImage image = ImageIO.read(input);
            int width = image.getWidth(), height = image.getHeight();
            double[][] R = new double[height][width];
            double[][] G = new double[height][width];
            double[][] B = new double[height][width];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int pixel = image.getRGB(x, y);
                    R[y][x] = ((pixel & 0xff0000) >> 16) / 255d;
                    G[y][x] = ((pixel & 0xff00) >> 8) / 255d;
                    B[y][x] = (pixel & 0xff) / 255d;
                }
            }
            return new double[][][]{R, G, B};
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void rgb2Image(Double[][][] pixels, String filePath) {
        try {
            int width = pixels[0].length, height = pixels[0][0].length;
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int r = (int) (pixels[0][y][x] * 255d) << 16;
                    int g = (int) (pixels[1][y][x] * 255d) << 8;
                    int b = (int) (pixels[2][y][x] * 255d);
                    image.setRGB(x, y, r + g + b);
                }
            }
            ImageIO.write(image, "JPG", new File(filePath));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void write(double[][] pixels, String filePath) {
        try {
            int width = pixels[0].length, height = pixels.length;
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int r = ((int) (pixels[y][x] * 255d)) << 16;
                    int g = ((int) (pixels[y][x] * 255d)) << 8;
                    int b = ((int) (pixels[y][x] * 255d));
                    image.setRGB(x, y, r + g + b);
                }
            }
            ImageIO.write(image, "JPG", new File(filePath));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void copy(String fileName, String target) {
        try {
            File file = new File(fileName);
            BufferedImage image = ImageIO.read(file);
            File outFile = new File(target);
            if (!outFile.exists()) outFile.mkdirs();
            ImageIO.write(image, "JPEG", new File(target + file.getName()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
