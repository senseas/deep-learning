package com.deep.framework.lang;

import com.deep.framework.lang.util.FileUtil;
import com.deep.framework.lang.util.ImageUtil;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.stream.IntStream;

public class DataLoader {
    public static final String BASE_PATH = System.getProperty("user.dir").concat("/");
    public static final String IMG_PATH = BASE_PATH.concat("img/");
    public static final String MODEL_PATH = BASE_PATH.concat("model/");
    public static final String TRAIN_IMAGES_FILE = "DataSet/train-images-idx3-ubyte";
    public static final String TRAIN_LABELS_FILE = "DataSet/train-labels-idx1-ubyte";
    public static final String TEST_IMAGES_FILE = "data/mnist/t10k-images.idx3-ubyte";
    public static final String TEST_LABELS_FILE = "data/mnist/t10k-labels.idx1-ubyte";
    public static final String[] TRIAN_IMAGES_140 = new String[]{
            "DataSet/a-140.jpg",
            "DataSet/b-140.jpg",
            "DataSet/c-140.jpg",
            "DataSet/d-140.jpg",
            "DataSet/e-140.jpg",
            "DataSet/f-140.jpg",
            "DataSet/g-140.jpg",
            "DataSet/h-140.jpg",
            "DataSet/k-140.jpg",
    };

   static  {
        if(!new File(IMG_PATH).exists()) new File(IMG_PATH).mkdir();
        if(!new File(MODEL_PATH).exists()) new File(MODEL_PATH).mkdir();
    }

    /**
     * get images of 'train' or 'test'
     * <p>
     * the file of 'train' or 'test' about image
     *
     * @return one row show a `picture`
     */
    public static double[][][][] getMnistImages() {
        double[][][][] images = null;
        InputStream input = FileUtil.readResourceAsStream(TRAIN_IMAGES_FILE);
        try (BufferedInputStream bin = new BufferedInputStream(input)) {
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
     * get labels of `train` or `test`
     * the file of 'train' or 'test' about label
     *
     * @return
     */
    public static double[][][] getMnistLabels() {
        double[][][] labes = null;
        InputStream input = FileUtil.readResourceAsStream(TRAIN_LABELS_FILE);
        try (BufferedInputStream bin = new BufferedInputStream(input)) {
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
        IntStream.range(0, TRIAN_IMAGES_140.length).forEach(i -> {
            input[i] = ImageUtil.image2RGB(TRIAN_IMAGES_140[i]);
        });
        return input;
    }

    public static void main(String[] args) {
        double[][][][] images = getMnistImages();
        IntStream.range(0, 2).forEach(i -> {
            String fileName = BASE_PATH.concat(String.valueOf(i)).concat(".JPEG");
            ImageUtil.write(images[i][0], fileName);
        });
    }

}
