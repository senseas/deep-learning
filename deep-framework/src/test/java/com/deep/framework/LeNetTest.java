package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tenser;
import com.deep.framework.graph.TensorFlow;
import org.apache.log4j.Logger;
import org.junit.Test;

public class LeNetTest extends Shape {
    Logger log = Logger.getLogger(LeNetTest.class);

    @Test
    public void LeNetTest() {

        Double[][][] inputSet = {
            {{0.1}, {0.1}}, {{0.1}, {0.2}}, {{0.2}, {0.2}}, {{0.2}, {0.3}}, {{0.3}, {0.7}}, {{0.4}, {0.8}}, {{0.5}, {0.9}}, {{0.8}, {0.9}}, {{0.6}, {0.8}}
        };
        Double[][][] labelSet = {{
            {0.01}}, {{0.02}}, {{0.04}}, {{0.06}}, {{0.21}}, {{0.32}}, {{0.45}}, {{0.72}}, {{0.48}}
        };

        TensorFlow tf = new TensorFlow();
        Tenser input = new Tenser(new int[]{3, 32, 32});
        Tenser label = new Tenser(new int[]{1, 1});

        Tenser tenser11 = tf.convx(new Tenser("weight", new int[]{6, 5, 5}), input);//6*28
        Tenser tenser12 = tf.relux(tenser11);//6*28
        Tenser tenser13 = tf.maxpool(tenser12);//6*14

        Tenser tenser21 = tf.convx(new Tenser("weight", new int[]{16, 5, 5}), tenser13);//16*10
        Tenser tenser22 = tf.relux(tenser21);//16*10
        Tenser tenser23 = tf.maxpool(tenser22);//16*5

        Tenser tenser31 = tf.convx(new Tenser("weight", new int[]{32, 5, 5}), tenser23);//32*1
        Tenser tenser32 = tf.matmul(new Tenser("weight", new int[]{86, 32}), tenser31);//86*1
        Tenser tenser33 = tf.addx(tenser32, new Tenser("bias", new int[]{86, 1}));//86*1
        Tenser tenser34 = tf.relux(tenser33);//86*1

        Tenser tenser41 = tf.matmul(new Tenser("weight", new int[]{32, 86}), tenser34);//32*86
        Tenser tenser42 = tf.addx(tenser41, new Tenser("bias", new int[]{32, 1}));
        Tenser tenser43 = tf.relux(tenser42);

        Tenser tenser51 = tf.matmul(new Tenser("weight", new int[]{32, 86}), tenser43);//32*86
        Tenser tenser52 = tf.addx(tenser51, new Tenser("bias", new int[]{32, 1}));//32
        Tenser tenser53 = tf.relux(tenser52);//32

        Tenser tenser61 = tf.matmul(new Tenser("weight", new int[]{10, 32}), tenser53);//32*86
        Tenser tenser62 = tf.addx(tenser61, new Tenser("bias", new int[]{10, 1}));//10
        Tenser tenser63 = tf.relux(tenser62);//10

        Tenser tenser71 = tf.softmax(tenser63);
        Tenser tenser72 = tf.squarex(label, tenser71);

        Executor executor = new Executor(tenser72, input, label);
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
