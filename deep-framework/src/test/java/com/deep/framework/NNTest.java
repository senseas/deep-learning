package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class NNTest extends Shape {

    @Test
    public void NNTest() {

        Double[][][] inputSet = {
            {{0.1}, {0.1}}, {{0.1}, {0.2}}, {{0.1}, {0.3}},
            {{0.2}, {0.2}}, {{0.2}, {0.3}}, {{0.2}, {0.5}},
            {{0.3}, {0.3}}, {{0.3}, {0.4}}, {{0.3}, {0.7}},
            {{0.4}, {0.5}}, {{0.4}, {0.6}}, {{0.4}, {0.8}},
            {{0.5}, {0.3}}, {{0.5}, {0.6}}, {{0.5}, {0.9}},
            {{0.8}, {0.2}}, {{0.8}, {0.7}}, {{0.8}, {0.9}},
            {{0.9}, {0.3}}, {{0.9}, {0.6}}, {{0.9}, {0.9}}
        };
        Double[][][] labelSet = {{
            {0.01}}, {{0.02}}, {{0.03}},
            {{0.04}}, {{0.06}},{{0.10}},
            {{0.09}}, {{0.12}}, {{0.21}},
            {{0.20}}, {{0.24}}, {{0.32}},
            {{0.15}}, {{0.30}}, {{0.45}},
            {{0.16}}, {{0.56}}, {{0.72}},
            {{0.27}}, {{0.45}}, {{0.81}},
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

        TensorExecutor executor = new TensorExecutor(tensor34, input, label);
        forEach(100000000, i -> {
            int l = (int) (Math.random() * labelSet.length);
            Object inSet = inputSet[l], labSet = labelSet[l];
            executor.run(inSet, labSet);
            if (i % 1000 == 0) {
                log.info("---------{}------------", i);
                Tensor loss = tensor34.getOutput().tensor();
                log("输入：", inSet);
                log("标签：", labSet);
                log("输出：", tensor33.data());
                log("误差：", loss.data());
            }
        });
    }

    public void log(String name, Object obj) {
        log.info(name.concat(JSONObject.toJSONString(obj)));
    }
}
