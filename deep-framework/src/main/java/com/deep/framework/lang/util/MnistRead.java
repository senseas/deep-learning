package com.deep.framework.lang.util;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.stream.IntStream;

public class MnistRead {
    public static final String BASE_PATH =System.getProperty("user.dir").concat("/src/main/resources/DataSet/");
    public static final String TRAIN_IMAGES_FILE = BASE_PATH.concat("train-images-idx3-ubyte");
    public static final String TRAIN_LABELS_FILE = BASE_PATH.concat("train-labels-idx1-ubyte");
    public static final String TEST_IMAGES_FILE = "data/mnist/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_FILE = "data/mnist/t10k-labels.idx1-ubyte";

    /**
     * change bytes into a hex string.
     *
     * @param bytes bytes
     * @return the returned hex string
     */
    public static String bytesToHex(byte[] bytes) {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < bytes.length; i++) {
            String hex = Integer.toHexString(bytes[i] & 0xFF);
            if (hex.length() < 2) {
                sb.append(0);
            }
            sb.append(hex);
        }
        return sb.toString();
    }

    /**
     * get images of 'train' or 'test'
     *
     * @param fileName the file of 'train' or 'test' about image
     * @return one row show a `picture`
     */
    public static double[][][][] getImages(String fileName) {
        double[][][][] images = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000803".equals(bytesToHex(bytes))) {                        // 读取魔数
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);      // 读取样本总数
                bin.read(bytes, 0, 4);
                int width = Integer.parseInt(bytesToHex(bytes), 16);       // 读取每行所含像素点数
                bin.read(bytes, 0, 4);
                int height = Integer.parseInt(bytesToHex(bytes), 16);      // 读取每列所含像素点数
                images = new double[number][1][][];
                for (int x = 0; x < number; x++) {
                    double[][] image = new double[height][width];
                    for (int i = 0; i < height; i++) {
                        for (int l = 0; l < width; l++) {
                            image[i][l] = bin.read()/255d;                      // 逐一读取像素值
                        }
                    }
                    images[x][0] = image;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return images;
    }

    /**
     * get labels of `train` or `test`
     *
     * @param fileName the file of 'train' or 'test' about label
     * @return
     */
    public static double[][][] getLabels(String fileName) {
        double[][][] labes = null;
        try (BufferedInputStream bin = new BufferedInputStream(new FileInputStream(fileName))) {
            byte[] bytes = new byte[4];
            bin.read(bytes, 0, 4);
            if (!"00000801".equals(bytesToHex(bytes))) {
                throw new RuntimeException("Please select the correct file!");
            } else {
                bin.read(bytes, 0, 4);
                int number = Integer.parseInt(bytesToHex(bytes), 16);
                labes = new double[number][10][1];
                for (int i = 0; i < number; i++) {
                    double[][] lab = {{0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}};
                    lab[bin.read()] = new double[]{1};
                    labes[i] = lab;
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return labes;
    }

    public static void drawGrayPicture(double[][] pixels, String fileName) {
        try {
            int width = pixels.length, height = pixels[0].length;
            BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            for (int i = 0; i < width; i++) {
                for (int l = 0; l < height; l++) {
                    int pixel = 255 - (int) pixels[i][l];
                    int value = pixel + (pixel << 8) + (pixel << 16);
                    bufferedImage.setRGB(l, i, value);
                }
            }
            ImageIO.write(bufferedImage, "JPEG", new File(fileName));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public static void main(String[] args) {
        System.out.println(ClassLoader.class.getResource(""));
        double[][][][] images = getImages(TRAIN_IMAGES_FILE);
        IntStream.range(0, 2).forEach(i -> {
            String fileName = BASE_PATH.concat(String.valueOf(i)).concat(".JPEG");
            drawGrayPicture(images[i][0], fileName);
        });
        double[][][] labels = getLabels(TRAIN_LABELS_FILE);
        System.out.println(labels);
    }
}
