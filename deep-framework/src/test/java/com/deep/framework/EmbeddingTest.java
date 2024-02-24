package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.util.Sequence;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static com.deep.framework.lang.util.Sequence.getTokenOneHotList;
import static com.deep.framework.lang.util.Sequence.getTokenIndex;

public class EmbeddingTest {
    int num = 28, dim = 13;

    @Test
    public void wordEmbeddingTest() {
        List<double[]> index = getTokenOneHotList("希望让普通读者也能轻易理解");
        double[] input = index.stream().flatMapToDouble(Arrays::stream).toArray();

        TensorFlow tf = new TensorFlow();
        Tensor tensor1 = new Tensor(input, new int[]{index.size(), num});
        Tensor tensor = tf.wordEmbedding(tensor1, new Tensor(new int[]{num, dim}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
        System.out.println(JSONObject.toJSONString(tensor));
    }

    @Test
    public void positionalEmbeddingTest() {
        double[] index = Sequence.getTokenIndex("希望让普通读者也能轻易理解");
        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(index, new int[]{index.length});
        Tensor tensor = tf.positionalEmbedding(new int[]{index.length, 13}, input);
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
        System.out.println(JSONObject.toJSONString(tensor));
    }
}