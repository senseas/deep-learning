package com.deep.framework;

import com.alibaba.fastjson2.JSONObject;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class CublasTest {

    @Test
    public void matmulTest() {
        Tensor input = new Tensor(new double[]{0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.2, 0.2}, new int[]{3, 3});
        Tensor weight = new Tensor(new double[]{0.01, 0.02, 0.03, 0.04, 0.06, 0.10}, new int[]{3, 2});

        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.matmul(input, weight);
        tensor.forward();
        tensor.setGrad(new double[]{0.2731f, 0.1389f, 0.7491f, -0.2307f, 0.3411f, 0.6492f});
        tensor.backward();

        System.out.println("     output:" + JSONObject.toJSONString(tensor.getData()));
        System.out.println(" input_grad:" + JSONObject.toJSONString(input.getGrad()));
        System.out.println("weight_grad:" + JSONObject.toJSONString(weight.getGrad()));
    }

    @Test
    public void matmulTranTest() {
        Tensor input = new Tensor(new double[]{0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.2, 0.2}, new int[]{3, 3});
        Tensor weight = new Tensor(new double[]{0.01, 0.02, 0.03, 0.04, 0.06, 0.10}, new int[]{2, 3});

        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.matmul(input, tf.matTran(weight));
        tensor.forward();
        tensor.setGrad(new double[]{0.2731f, 0.1389f, 0.7491f, -0.2307f, 0.3411f, 0.6492f});
        tensor.backward();

        System.out.println("     output:" + JSONObject.toJSONString(tensor.getData()));
        System.out.println(" input_grad:" + JSONObject.toJSONString(input.getGrad()));
        System.out.println("weight_grad:" + JSONObject.toJSONString(weight.getGrad()));
    }

    @Test
    public void matmulTran1Test() {
        Tensor input = new Tensor(new double[]{0.1, 0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.2, 0.2}, new int[]{3, 3});
        Tensor weight = new Tensor(new double[]{0.01, 0.02, 0.03, 0.04, 0.06, 0.10}, new int[]{2, 3});

        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.matmulTran(input, weight);
        tensor.forward();
        tensor.setGrad(new double[]{0.2731f, 0.1389f, 0.7491f, -0.2307f, 0.3411f, 0.6492f});
        tensor.backward();

        System.out.println("     output:" + JSONObject.toJSONString(tensor.getData()));
        System.out.println(" input_grad:" + JSONObject.toJSONString(input.getGrad()));
        System.out.println("weight_grad:" + JSONObject.toJSONString(weight.getGrad()));
    }
}