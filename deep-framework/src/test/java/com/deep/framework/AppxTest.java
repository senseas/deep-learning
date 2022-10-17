package com.deep.framework;

import com.deep.framework.framework.TensorExecutor;
import com.deep.framework.framework.TensorFlow;
import com.deep.framework.graph.Tensor;
import org.junit.Test;

public class AppxTest {

    @Test
    public void squarexTest() {
        TensorFlow tf = new TensorFlow();
        Tensor softmax = tf.softmaxCrossx(new Tensor(new int[]{2}), new Tensor(new int[]{2}));
        TensorExecutor executor = new TensorExecutor(softmax);
        executor.run();
    }

}
