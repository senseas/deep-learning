package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFlow;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.DataLoader;
import com.deep.framework.lang.util.ImageUtil;
import com.deep.framework.lang.ModeLoader;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeMap;
import java.util.stream.IntStream;

@Slf4j
public class LeNetTest extends Shape {

    @Test
    public void LeNetTest() {

        double[][][][] inputSet = DataLoader.getMnistImages(DataLoader.TRAIN_IMAGES_FILE);
        double[][][] labelSet = DataLoader.getMnistLabels(DataLoader.TRAIN_LABELS_FILE);

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{1, 28, 28});
        Tensor label = new Tensor(new int[]{10, 1});

        Tensor tensor11 = tf.convx(new Tensor("weight", new int[]{10, 5, 5}), input);//6*24
        Tensor tensor12 = tf.relux(tensor11);//6*24
        Tensor tensor13 = tf.maxpoolx(tensor12);//6*12

        Tensor tensor21 = tf.convx(new Tensor("weight", new int[]{16, 5, 5}), tensor13);//16*8
        Tensor tensor22 = tf.relux(tensor21);//16*8
        Tensor tensor23 = tf.maxpoolx(tensor22);//16*4

        Tensor tensor31 = tf.convx(new Tensor("weight", new int[]{32, 4, 4}), tensor23);//32*1
        Tensor tensor30 = tf.shape(tensor31, new Tensor(new int[]{32, 1}));
        Tensor tensor32 = tf.matmul(new Tensor("weight", new int[]{86, 32}), tensor30);//86*1
        Tensor tensor33 = tf.addx(tensor32, new Tensor("bias", new int[]{86, 1}));//86*1
        Tensor tensor34 = tf.relux(tensor33);//86*1

        Tensor tensor41 = tf.matmul(new Tensor("weight", new int[]{32, 86}), tensor34);//32*86
        Tensor tensor42 = tf.addx(tensor41, new Tensor("bias", new int[]{32, 1}));
        Tensor tensor43 = tf.relux(tensor42);

        Tensor tensor51 = tf.matmul(new Tensor("weight", new int[]{10, 32}), tensor43);//32*86
        Tensor tensor52 = tf.addx(tensor51, new Tensor("bias", new int[]{10, 1}));//10
        Tensor tensor53 = tf.sigmoidx(tensor52);//10

        Tensor softmax = tf.softmax(tensor53);
        Tensor<None> crossx = tf.softmaxCrossx(label, softmax);

        Executor executor = new Executor(crossx, input, label);
        forEach(20, x -> {
            forEach(60000, i -> {
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                if (i % 500 == 0) {
                    log.info("---------{}------------", i);
                    ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                    log(Shape.reshape(labSet, new Double[10]));
                    log(Shape.reshape(softmax.getOutput(), new None[10]));
                    log(crossx.getOutput());
                }
            });
        });
    }

    @Test
    public void TrainTest() {
        double[][][][] inputSet = DataLoader.getMnistImages(DataLoader.TRAIN_IMAGES_FILE);
        double[][][] labelSet = DataLoader.getMnistLabels(DataLoader.TRAIN_LABELS_FILE);

        Executor executor = ModeLoader.load(DataLoader.BASE_PATH.concat("LetNet.obj"));
        Tensor<None> crossx = executor.getTensor();
        Tensor softmax = crossx.getInput()[1];
        forEach(20, x -> {
            forEach(60000, i -> {
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                if (i % 500 == 0) {
                    log.info("---------{}------------", i);
                    ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                    log(Shape.reshape(labSet, new Double[10]));
                    log(Shape.reshape(softmax.getOutput(), new None[10]));
                    log(crossx.getOutput());
                }
            });
        });
    }

    @Test
    public void EvalTest() {
        double[][][][] inputSet = DataLoader.getMnistImages(DataLoader.TRAIN_IMAGES_FILE);
        double[][][] labelSet = DataLoader.getMnistLabels(DataLoader.TRAIN_LABELS_FILE);

        Executor executor = ModeLoader.load(DataLoader.BASE_PATH.concat("LetNet.obj"));
        Tensor<None> crossx = executor.getTensor();
        Tensor softmax = crossx.getInput()[1];
        List list = new ArrayList();
        forEach(60000, i -> {
            log.info("---------{}------------", i);
            Object inSet = inputSet[i], labSet = labelSet[i];
            executor.forward(inSet, labSet);
            Double[] label = Shape.reshape(labSet, new Double[10]);
            None[] output = Shape.reshape(softmax.getOutput(), new None[10]);

            double sum = IntStream.range(0, 9).mapToDouble(a -> a * label[a]).sum();
            TreeMap<Double, Integer> map = new TreeMap<>(Comparator.reverseOrder());
            IntStream.range(0, 9).forEach(a -> map.put(output[a].getValue(), a));
            if (sum == map.get(map.firstKey())) list.add(output);
            log.info("标签：  {}", sum);
            log.info("输出：  {}", JSONObject.toJSONString(map));
            log.info("识别率: {}", ((double) list.size()) / (i == 0 ? 1 : i));
            log(crossx.getOutput());
        });
    }

    @Test
    public void ImgTest() {
        int index = 12500;
        double[][][][] images = DataLoader.getMnistImages(DataLoader.TRAIN_IMAGES_FILE);
        String fileName = DataLoader.BASE_PATH.concat(String.valueOf(index)).concat(".JPEG");
        ImageUtil.write(images[index][0], fileName);
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
