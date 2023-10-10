package com.deep.framework;

import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import org.junit.Test;

public class AppTest {

    @Test
    public void sigmoidTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.sigmoid(new Tensor(-0.6354469361189982));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();

        Double value = 1 / (1 + Math.exp(-(-0.6354469361189982)));
        System.out.println(value);
        Double value1 = value * (1 - value);
        System.out.println(value1 * 0.1694231856183997);
    }

    @Test
    public void sigmoidTestxx() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.sigmoid(new Tensor(-0.6354469361189982));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
        Tensor none = tensor.getInput()[0].getOutput().tensor();
        //gradient(tensor);
        System.out.println(none.grad());

        Double value = 1 / (1 + Math.exp(-(-0.6354469361189982)));
        System.out.println(value);
        Double value1 = value * (1 - value);
        System.out.println(value1 * 0.1694231856183997);
    }

    @Test
    public void reluxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.relux(new Tensor(new int[]{2,5}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
        Tensor none = tensor.getInput()[0].getOutput().tensor();
        //compute(tensor);
        System.out.println(none);
    }

    @Test
    public void appTest() {
        TensorFlow tf = new TensorFlow();
        Tensor x = new Tensor(2d);
        Tensor m = tf.mul(tf.minus(new Tensor(6d), x), x);
        TensorExecutor executor = new TensorExecutor(m);
        executor.run();
    }

    @Test
    public void matmulTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.matmul(new Tensor(new int[]{6, 4}), new Tensor(new int[]{4, 1}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

    @Test
    public void squareTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.square(new Tensor(0.01), new Tensor(0.391249035007275));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }
    @Test
    public void squarexTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.squarex(new Tensor(new int[]{6}), new Tensor(new int[]{6}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

    @Test
    public void softmaxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.softmax(new Tensor(new int[]{2}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

    @Test
    public void convTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.conv(new int[]{1, 1}, new int[]{0, 0}, new Tensor(new int[]{5, 5}), new Tensor(new int[]{140, 140}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

    @Test
    public void convxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.convx(new int[]{1, 1}, new int[]{0, 0}, new Tensor(new int[]{10, 5, 5}), new Tensor(new int[]{3, 140, 140}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

    @Test
    public void deconvTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.deconv(new int[]{1, 1}, new int[]{0, 0}, new Tensor(new int[]{5, 5}), new Tensor(new int[]{140, 140}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

    @Test
    public void demaxpoolTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.demaxpool(2, new int[]{2, 2}, new int[]{0, 0}, new Tensor(new int[]{140, 140}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

    @Test
    public void batchNormTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.batchNorm(new Tensor(new int[]{2, 2}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
    }

}
