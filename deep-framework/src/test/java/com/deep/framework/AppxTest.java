package com.deep.framework;

import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import org.junit.Test;

public class AppxTest {

    @Test
    public void softmaxTest() {
        TensorFlow tf = new TensorFlow();
        double[] floats = {0.2731f, 0.1389f, 0.7491f, 0.2307f, 0.3411f, 0.6492f, 0.2313f, 0.5270f, 0.6267f, 0.2598f};
        Tensor softmax = tf.softmax(new Tensor(floats, new int[]{10}));
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
