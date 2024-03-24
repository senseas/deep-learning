package com.deep.framework;

import com.deep.framework.functions.Tensor;
import com.deep.framework.functions.TensorFlow;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class FunctionsTest {

    @Test
    public void layerNormalTest() {
        TensorFlow tf = new TensorFlow();
        Tensor data1 = new Tensor(new int[]{3, 1});
        Tensor data2 = new Tensor(new int[]{3, 1});
        Tensor data3 = new Tensor(new int[]{3, 1});
        Tensor layerNormal = tf.layerNormal(data1, data2, data3);

        layerNormal.forward();
        layerNormal.getOutput().forEach((Tensor out, int i) -> {
            Tensor grad = new Tensor("g" + layerNormal.getId() + "[" + i + "]");
            out.setGrad(grad);
        });
        layerNormal.backward();
        layerNormal.getInput()[0].getOutput().forEach((out) -> {
            Tensor grad = out.getGrad();
            List<String> list = new ArrayList<>();
            while (true) {
                grad.reducer();
                if (list.contains(grad.getData())) return;
                list.add(grad.getData());
            }
        });
    }

}