package com.deep.framework;

import com.deep.framework.framework.Executor;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFlow;
import org.junit.Test;

public class AppTest {

    @Test
    public void sigmoidTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.sigmoid(new Tensor(-0.6354469361189982));
        Executor executor = new Executor(tensor);
        executor.run();

        Double value = 1 / (1 + Math.exp(-(-0.6354469361189982)));
        System.out.println(value);
        Double value1 = value * (1 - value);
        System.out.println(value1 * 0.1694231856183997);
    }

    @Test
    public void appaTest() {
        TensorFlow tf = new TensorFlow();
        Tensor x = new Tensor(2d);
        Tensor m = tf.mul(tf.minus(new Tensor(6d), x), x);
        Executor executor = new Executor(m);
        executor.run();
    }

    @Test
    public void matmulTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.matmul(new Tensor(new int[]{6, 4}), new Tensor(new int[]{4, 1}));
        Executor executor = new Executor(tensor);
        executor.run();
    }

    @Test
    public void squarexTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.square(new Tensor(0.01), new Tensor(0.391249035007275));
        Executor executor = new Executor(tensor);
        executor.run();
    }

    @Test
    public void softmaxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.softmax(new Tensor(new int[]{2}));
        Executor executor = new Executor(tensor);
        executor.run();
    }

    @Test
    public void cnnTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.conv(new Tensor(new int[]{5, 5}), new Tensor(new int[]{140, 140}));
        Executor executor = new Executor(tensor);
        executor.run();
    }

    @Test
    public void cnnxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor tensor = tf.convx(new Tensor(new int[]{10, 5, 5}), new Tensor(new int[]{3, 140, 140}));
        Executor executor = new Executor(tensor);
        executor.run();
    }

}
