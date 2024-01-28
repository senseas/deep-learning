package com.deep.framework;

import com.deep.framework.functions.Tensor;
import com.deep.framework.functions.TensorFlow;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

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
        forEach(layerNormal.getOutput(), (Tensor out) -> {
            Tensor grad = new Tensor();
            grad.setData(grad.getGradId());
            out.setGrad(grad);
        });
        layerNormal.backward();

        forEach(layerNormal.getInput()[0].getOutput(), (Tensor out) -> {
            Tensor grad = out.getGrad();
            List<String> list = new ArrayList<>();
            while (true) {
                grad.reducer();
                if(list.contains(grad.getData())) return;
                list.add(grad.getData());
            }
        });
    }

}