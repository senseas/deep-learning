package com.deep.framework.lang;

import com.deep.framework.lang.util.ImageUtil;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.stream.IntStream;

public class DataLoader {
    public static final String BASE_PATH = System.getProperty("user.dir").concat("/src/main/resources/DataSet/");
    public static final String TRAIN_IMAGES_FILE = BASE_PATH.concat("train-images-idx3-ubyte");
    public static final String TRAIN_LABELS_FILE = BASE_PATH.concat("train-labels-idx1-ubyte");
    public static final String TEST_IMAGES_FILE = "data/mnist/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_FILE = "data/mnist/t10k-labels.idx1-ubyte";
    public static final String[] TRIAN_IMAGES_140 = new String[]{
        BASE_PATH.concat("a-140.jpg"),
        BASE_PATH.concat("b-140.jpg"),
        BASE_PATH.concat("k-140.jpg"),
    };

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
    public static double[][][][] getMnistImages(String fileName) {
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
                            image[i][l] = bin.read() / 255d;                      // 逐一读取像素值
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
    public static double[][][] getMnistLabels(String fileName) {
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

    public static double[][][][] getImageData() {
        double[][][][] input = new double[TRIAN_IMAGES_140.length][][][];
        IntStream.range(0,TRIAN_IMAGES_140.length).forEach(i->{
            input[i] = ImageUtil.image2RGB(TRIAN_IMAGES_140[i]);
        });
        return input;
    }

    public static void main(String[] args) {
        System.out.println(ClassLoader.class.getResource(""));
        double[][][][] images = getMnistImages(TRAIN_IMAGES_FILE);
        IntStream.range(0, 2).forEach(i -> {
            String fileName = BASE_PATH.concat(String.valueOf(i)).concat(".JPEG");
            ImageUtil.write(images[i][0], fileName);
        });
    }

}
