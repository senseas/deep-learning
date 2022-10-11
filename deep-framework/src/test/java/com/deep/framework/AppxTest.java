package com.deep.framework;

import com.deep.framework.framework.TensorCore;
import com.deep.framework.framework.TensorFlow;
import com.deep.framework.graph.Tensor;
import org.junit.Test;

public class AppxTest {

    @Test
    public void squarexTest() {
        TensorFlow tf = new TensorFlow();
        Tensor softmax = tf.sigmoid(new Tensor(-0.6354469361189982));
        softmax.forward();
        TensorCore.forward(softmax);
        System.out.println(TensorCore.fparam);
        String code = TensorCore.code;
        System.out.println(code);

        TensorCore.backward(softmax);
    }

}
