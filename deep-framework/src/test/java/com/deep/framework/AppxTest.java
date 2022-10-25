package com.deep.framework;

import com.deep.framework.framework.TensorExecutor;
import com.deep.framework.framework.TensorFlow;
import com.deep.framework.graph.Tensor;
import org.junit.Test;

public class AppxTest {

    @Test
    public void softmaxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor softmax = tf.softmax(new Tensor(new int[]{3}));
        TensorExecutor executor = new TensorExecutor(softmax);
        executor.run();
    }

    @Test
    public void sigmoidxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor sigmoidx = tf.sigmoidx(new Tensor(new int[]{3}));
        TensorExecutor executor = new TensorExecutor(sigmoidx);
        executor.run();
    }

    @Test
    public void softmaxCrossxTest() {
        TensorFlow tf = new TensorFlow();
        Tensor softmaxCrossx = tf.softmaxCrossx(new Tensor(new int[]{2}),new Tensor(new int[]{2}));
        TensorExecutor executor = new TensorExecutor(softmaxCrossx);
        executor.run();
        executor.run();
    }

}
