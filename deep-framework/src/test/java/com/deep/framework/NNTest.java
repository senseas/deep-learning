package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Shape;
import com.deep.framework.graph.Tensor;
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
        Tensor input = new Tensor(new int[]{2, 1});
        Tensor label = new Tensor(new int[]{1, 1});

        Tensor tensor11 = tf.matmul(new Tensor("weight", new int[]{4, 2}), input);
        Tensor tensor12 = tf.addx(tensor11, new Tensor("bias", new int[]{4, 1}));
        Tensor tensor13 = tf.sigmoidx(tensor12);

        Tensor tensor21 = tf.matmul(new Tensor("weight", new int[]{6, 4}), tensor13);
        Tensor tensor22 = tf.addx(tensor21, new Tensor("bias", new int[]{6, 1}));
        Tensor tensor23 = tf.sigmoidx(tensor22);

        Tensor tensor31 = tf.matmul(new Tensor("weight", new int[]{1, 6}), tensor23);
        Tensor tensor32 = tf.addx(tensor31, new Tensor("bias", new int[]{1, 1}));
        Tensor tensor33 = tf.sigmoidx(tensor32);
        Tensor tensor34 = tf.squarex(label, tensor33);

        Executor executor = new Executor(tensor34, input, label);
        forEach(100000000, i -> {
            int l = (int) (Math.random() * labelSet.length);
            Object inSet = inputSet[l], labSet = labelSet[l];

            executor.run(inSet, labSet);
            if (i % 1000 == 0) {
                if (executor.rate > 0.00001) executor.rate = executor.rate - 0.0001;
                log.info("---------{" + i + "}------------");
                None[][] output = (None[][]) tensor33.getOutput();
                None loss = (None) tensor34.getOutput();
                log("输入：", inSet);
                log("标签：", labSet);
                log("输出：", output[0][0].getValue());
                log("误差：", loss.getValue());
            }
        });
    }

    public void log(String name, Object obj) {
        log.info(name.concat(JSONObject.toJSONString(obj)));
    }
}
