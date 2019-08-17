package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tenser;
import com.deep.framework.graph.TensorFlow;
import org.apache.log4j.Logger;
import org.junit.Test;

public class NNTest extends Shape {
    Logger log = Logger.getLogger(NNTest.class);

    @Test
    public void NNTest() {

        Double[][][] inputSet = {
                {{0.1}, {0.1}}, {{0.1}, {0.2}}, {{0.2}, {0.2}}, {{0.2}, {0.3}}, {{0.3}, {0.7}}, {{0.4}, {0.8}}, {{0.5}, {0.9}}, {{0.8}, {0.9}}, {{0.6}, {0.8}}
        };
        Double[][][] labelSet = {{
                {0.01}}, {{0.02}}, {{0.04}}, {{0.06}}, {{0.21}}, {{0.32}}, {{0.45}}, {{0.72}}, {{0.48}}
        };

        TensorFlow tf = new TensorFlow();
        Tenser input = new Tenser(new int[]{2, 1});
        Tenser label = new Tenser(new int[]{1, 1});

        Tenser tenser11 = tf.matmul(new Tenser("weight", new int[]{4, 2}), input);
        Tenser tenser12 = tf.addx(tenser11, new Tenser("bias", new int[]{4, 1}));
        Tenser tenser13 = tf.sigmoidx(tenser12);

        Tenser tenser21 = tf.matmul(new Tenser("weight", new int[]{6, 4}), tenser13);
        Tenser tenser22 = tf.addx(tenser21, new Tenser("bias", new int[]{6, 1}));
        Tenser tenser23 = tf.sigmoidx(tenser22);

        Tenser tenser31 = tf.matmul(new Tenser("weight", new int[]{1, 6}), tenser23);
        Tenser tenser32 = tf.addx(tenser31, new Tenser("bias", new int[]{1, 1}));
        Tenser tenser33 = tf.sigmoidx(tenser32);
        Tenser tenser34 = tf.squarex(label, tenser33);

        Executor executor = new Executor(tenser34, input, label);
        forEach(100000000, i -> {
            int l = (int) (Math.random() * labelSet.length);
            Object inSet = inputSet[l], labSet = labelSet[l];

            executor.run(inSet, labSet);
            if (i % 1000 == 0) {
                if (executor.rate > 0.00001)
                    executor.rate = executor.rate - 0.0001;
                log.info("---------{" + i + "}------------");
                log(tenser33.getOutput());
                log(tenser34);
            }
        });
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
