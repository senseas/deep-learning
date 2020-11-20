package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFlow;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.util.MnistUtil;
import com.deep.framework.lang.util.ModelUtil;
import org.apache.log4j.Logger;
import org.junit.Test;

import java.util.stream.IntStream;

public class LeNetTest extends Shape {
    Logger log = Logger.getLogger(LeNetTest.class);

    @Test
    public void LeNetTest() {

        double[][][][] inputSet = MnistUtil.getImages(MnistUtil.TRAIN_IMAGES_FILE);
        double[][][] labelSet = MnistUtil.getLabels(MnistUtil.TRAIN_LABELS_FILE);

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
        Tensor tensor53 = tf.relux(tensor52);//10

        Tensor softmax = tf.softmax(tensor53);
        Tensor<None> crossx = tf.crossx(label, softmax);

        Executor executor = new Executor(crossx, input, label);
        forEach(3, x -> {
            forEach(60000, i -> {
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet, o -> {
                    if (i % 100 == 0) {
                        log.info("---------{" + i + "}------------");
                        ModelUtil.save(executor, MnistUtil.BASE_PATH.concat("LetNet.obj"));
                        log(Shape.reshape(labSet, new Double[10]));
                        log(Shape.reshape(softmax.getOutput(), new None[10]));
                        log(crossx.getOutput());
                    }
                });
            });
        });
    }

    @Test
    public void TrainTest() {
        double[][][][] inputSet = MnistUtil.getImages(MnistUtil.TRAIN_IMAGES_FILE);
        double[][][] labelSet = MnistUtil.getLabels(MnistUtil.TRAIN_LABELS_FILE);

        Executor executor = ModelUtil.load(MnistUtil.BASE_PATH.concat("LetNet.obj"));
        Tensor<None> crossx = executor.getTensor();
        Tensor softmax = crossx.getInput()[1];
        forEach(60000, i -> {
            Object inSet = inputSet[i], labSet = labelSet[i];
            executor.run(inSet, labSet);
            if (i % 100 == 0) {
                log.info("---------{" + i + "}------------");
                ModelUtil.save(executor, MnistUtil.BASE_PATH.concat("LetNet.obj"));
                log(Shape.reshape(labSet, new Double[10]));
                log(Shape.reshape(softmax.getOutput(), new None[10]));
                log(crossx.getOutput());
            }
        });
    }

    @Test
    public void imgTest() {
        double[][][][] images = MnistUtil.getImages(MnistUtil.TRAIN_IMAGES_FILE);
        IntStream.range(34500, 34500 + 1).forEach(i -> {
            String fileName = MnistUtil.BASE_PATH.concat(String.valueOf(i)).concat(".JPEG");
            MnistUtil.drawGrayPicture(images[i][0], fileName);
        });
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
