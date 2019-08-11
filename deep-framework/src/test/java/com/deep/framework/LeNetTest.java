package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tenser;
import com.deep.framework.graph.TensorFlow;
import com.deep.framework.lang.util.MnistRead;
import org.apache.log4j.Logger;
import org.junit.Test;

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
        Tenser tenser30 = new Tenser(Shape.reshape(tenser31.getFunction(), new Tenser[32][1]));
        Tenser tenser32 = tf.matmul(new Tenser("weight", new int[]{86, 32}), tenser30);//86*1
        Tenser tenser33 = tf.addx(tenser32, new Tenser("bias", new int[]{86, 1}));//86*1
        Tenser tenser34 = tf.relux(tenser33);//86*1

        Tenser tenser41 = tf.matmul(new Tenser("weight", new int[]{32, 86}), tenser34);//32*86
        Tenser tenser42 = tf.addx(tenser41, new Tenser("bias", new int[]{32, 1}));
        Tenser tenser43 = tf.relux(tenser42);

        Tenser tenser51 = tf.matmul(new Tenser("weight", new int[]{10, 32}), tenser43);//32*86
        Tenser tenser52 = tf.addx(tenser51, new Tenser("bias", new int[]{10, 1}));//10
        Tenser tenser53 = tf.relux(tenser52);//10

        Tenser tenser61 = tf.softmax(tenser53);
        Tenser tenser62 = tf.crossx(label, tenser61);

        Executor executor = new Executor(tenser62, input, label);
        forEach(100000000, i -> {
            int l = (int) (Math.random() * labelSet.length);
            Object inSet = inputSet[l], labSet = labelSet[l];

            executor.run(inSet, labSet);
            if (i % 1000 == 0) {
                if (executor.rate > 0.00001)
                    executor.rate = executor.rate - 0.0001;
                log.info("---------{" + i + "}------------");
                log(tenser61.getOutput());
                log(tenser62);
            }
        });
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
