package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.bean.None;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tenser;
import com.deep.framework.graph.TensorFlow;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.util.MnistRead;
import org.apache.log4j.Logger;
import org.junit.Test;

import java.util.stream.IntStream;

public class LeNetTest extends Shape {
    Logger log = Logger.getLogger(LeNetTest.class);

    @Test
    public void LeNetTest() {

        double[][][][] inputSet = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        double[][][] labelSet = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);

        TensorFlow tf = new TensorFlow();
        Tenser input = new Tenser(new int[]{1, 28, 28});
        Tenser label = new Tenser(new int[]{10, 1});

        Tenser tenser11 = tf.convx(new Tenser("weight", new int[]{6, 5, 5}), input);//6*24
        Tenser tenser12 = tf.relux(tenser11);//6*24
        Tenser tenser13 = tf.maxpoolx(tenser12);//6*12

        Tenser tenser21 = tf.convx(new Tenser("weight", new int[]{16, 5, 5}), tenser13);//16*8
        Tenser tenser22 = tf.relux(tenser21);//16*8
        Tenser tenser23 = tf.maxpoolx(tenser22);//16*4

        Tenser tenser31 = tf.convx(new Tenser("weight", new int[]{32, 4, 4}), tenser23);//32*1
        Tenser tenser30 = tf.shape(tenser31, new Tenser(new int[]{32, 1}));
        Tenser tenser32 = tf.matmul(new Tenser("weight", new int[]{86, 32}), tenser30);//86*1
        Tenser tenser33 = tf.addx(tenser32, new Tenser("bias", new int[]{86, 1}));//86*1
        Tenser tenser34 = tf.relux(tenser33);//86*1

        Tenser tenser41 = tf.matmul(new Tenser("weight", new int[]{32, 86}), tenser34);//32*86
        Tenser tenser42 = tf.addx(tenser41, new Tenser("bias", new int[]{32, 1}));
        Tenser tenser43 = tf.relux(tenser42);

        Tenser tenser51 = tf.matmul(new Tenser("weight", new int[]{10, 32}), tenser43);//32*86
        Tenser tenser52 = tf.addx(tenser51, new Tenser("bias", new int[]{10, 1}));//10
        Tenser tenser53 = tf.relux(tenser52);//10

        Tenser softmax = tf.softmax(tenser53);
        Tenser<None> crossx = tf.crossx(label, softmax);

        Executor executor = new Executor(crossx, input, label);
        forEach(60000, i -> {
            Object inSet = inputSet[i], labSet = labelSet[i];
            Func1 func = o -> executor.rate = crossx.getOutput().getValue() / 1000;
            executor.run(inSet, labSet, func);
            if (i % 100 == 0) {
                log.info("---------{" + i + "}------------");
                saveModel(executor, MnistRead.BASE_PATH.concat("LetNet.obj"));
                log(labSet);
                log(softmax.getOutput());
                log(crossx.getOutput());
            }
        });
    }

    @Test
    public void TrainTest() {
        double[][][][] inputSet = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        double[][][] labelSet = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);

        Executor executor = loadModel(MnistRead.BASE_PATH.concat("LetNet.obj"));
        Tenser<None> crossx = executor.getTenser();
        Tenser softmax = (Tenser) crossx.getInput()[1];
        forEach(60000, i -> {
            Object inSet = inputSet[i], labSet = labelSet[i];
            Func1 func = o -> executor.rate = crossx.getOutput().getValue() / 1000;
            executor.run(inSet, labSet, func);
            if (i % 100 == 0) {
                log.info("---------{" + i + "}------------");
                saveModel(executor, MnistRead.BASE_PATH.concat("LetNet.obj"));
                log(labSet);
                log(softmax.getOutput());
                log(crossx.getOutput());
            }
        });
    }

    @Test
    public void imgTest() {
        double[][][][] images = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        IntStream.range(37426, 37426 + 1).forEach(i -> {
            String fileName = MnistRead.BASE_PATH.concat(String.valueOf(i)).concat(".JPEG");
            MnistRead.drawGrayPicture(images[i][0], fileName);
        });
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
