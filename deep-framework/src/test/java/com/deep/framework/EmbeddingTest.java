package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import com.huaban.analysis.jieba.JiebaSegmenter;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class EmbeddingTest {
    private static final JiebaSegmenter jieba_segmenter = new JiebaSegmenter();
    String str = "在本文中，我们将试图把模型简化一点，并逐一介绍里面的核心概念，希望让普通读者也能轻易理解。";

    int num = 28, dim = 13;
    List<String> wordTable = getWordTable(str);

    @Test
    public void embeddingTest() {
        List<double[]> index = getWordList("希望让普通读者也能轻易理解");
        double[] input = index.stream().flatMapToDouble(Arrays::stream).toArray();

        TensorFlow tf = new TensorFlow();
        Tensor tensor1 = new Tensor(input, new int[]{index.size(), num});
        Tensor tensor = tf.matmul(tensor1, new Tensor(new int[]{num, dim}));
        TensorExecutor executor = new TensorExecutor(tensor);
        executor.run();
        System.out.println(JSONObject.toJSONString(tensor));
    }

    private List<double[]> getWordList(String str) {
        return jieba_segmenter
            .process(str, JiebaSegmenter.SegMode.INDEX)
            .stream().map(a -> oneHot(wordTable.indexOf(a.word)))
            .toList();
    }

    private List<String> getWordTable(String str) {
        return jieba_segmenter
            .process(str, JiebaSegmenter.SegMode.INDEX)
            .stream().map(a -> a.word)
            .distinct().sorted().toList();
    }

    private double[] oneHot(int idx) {
        return IntStream.range(0, num)
            .mapToDouble(a -> a == idx ? 1 : 0)
            .toArray();
    }

}
