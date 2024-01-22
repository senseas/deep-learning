package com.deep.framework;

import com.deep.framework.functions.Tensor;
import com.deep.framework.functions.TensorFlow;
import org.junit.Test;

import static com.deep.framework.lang.ForEach.forEach;

public class FunctionsTest {

    @Test
    public void layerNormalTest() {
        TensorFlow tf = new TensorFlow();
        Tensor data1 = new Tensor(new int[]{3, 1});
        Tensor data2 = new Tensor(new int[]{3, 1});
        Tensor data3 = new Tensor(new int[]{3, 1});
        Tensor layerNormal = tf.layerNormal(data1, data2, data3);

        layerNormal.forward();
        forEach(layerNormal.getOutput(), (Tensor out) -> out.setGrad(new Tensor(System.nanoTime() + "")));
        layerNormal.backward();
        Tensor.reduces = true;

        forEach(layerNormal.getInput()[0].getOutput(), (Tensor out) -> {
            Tensor grad = out.getGrad();
            grad.reducer();
            grad.reducer();
            grad.reducer();
            grad.reducer();
            grad.reducer();
            grad.reducer();
        });
    }

}