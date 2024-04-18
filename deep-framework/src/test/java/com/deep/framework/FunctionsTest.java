package com.deep.framework;

import com.deep.framework.functions.Tensor;
import com.deep.framework.functions.TensorExecutor;
import com.deep.framework.functions.TensorFlow;
import org.junit.Test;

public class FunctionsTest {

    @Test
    public void layerNormalTest() {
        TensorFlow tf = new TensorFlow();
        Tensor data1 = new Tensor(new int[]{3, 1});
        Tensor data2 = new Tensor(new int[]{3, 1});
        Tensor data3 = new Tensor(new int[]{3, 1});
        Tensor layerNormal = tf.layerNormal(data1, data2, data3);
        new TensorExecutor(layerNormal).run();
    }

    @Test
    public void softmaxCrossTest() {
        TensorFlow tf = new TensorFlow();
        Tensor data1 = new Tensor();
        Tensor data2 = new Tensor();
        Tensor softmaxCross = tf.softmaxCross(data1, data2);
        new TensorExecutor(softmaxCross).run();
    }

}