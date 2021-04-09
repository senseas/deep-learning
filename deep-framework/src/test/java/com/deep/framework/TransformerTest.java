package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFlow;
import com.deep.framework.lang.Shape;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class TransformerTest extends Shape {

    @Test
    public void TransformerTest() {
        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{ 64, 64});
        Tensor label = new Tensor(new int[]{10, 1});
        Tensor tensor = tf.selfAttention(input, new Tensor(new int[]{3, 64, 512}));
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
