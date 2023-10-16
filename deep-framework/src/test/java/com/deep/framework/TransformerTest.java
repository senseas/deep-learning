package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.Arrays;

import static com.deep.framework.lang.ForEach.forEach;
import static com.deep.framework.lang.util.Sequence.getWordIndexList;
import static com.deep.framework.lang.util.Sequence.oneHot;

@Slf4j
public class TransformerTest{

    /**
     * gpt decoder
     * 模型初步搭建
     */
    @Test
    public void transformerTest() {
        int num = 28, dim = 13;
        TensorFlow tf = new TensorFlow();
        double[] words = getWordIndexList("<begin>10月13日，@京东发言人 发文称，我们关注到有谣言称“刘姓商人涉嫌违法被抓”，该谣言被别有用心的人刻意发布在京东相关新闻动态下，以混淆视听、操纵舆论。我们对此恶劣行径表示强烈愤慨，并已向公安机关报案。");
        double[] data = Arrays.stream(words).mapToObj(a -> oneHot((int) a)).flatMapToDouble(Arrays::stream).toArray();
        //Embedding
        Tensor tensor11 = new Tensor(data, new int[]{words.length, dim});
        Tensor tensor12 = tf.matmul(tensor11, new Tensor(new int[]{num, dim}));
        Tensor tensor13 = tf.positionalEmbedding(new int[]{words.length, dim}, new Tensor(words, new int[]{words.length}));
        Tensor tensor14 = tf.addx(tensor12, tensor13);

        //MultiHeadAttention
        Tensor tensor15 = tf.multiHeadAttention(8, tensor14, new Tensor(new int[]{8, 3, num, dim}), new Tensor(new int[]{64, 64}), new Tensor(new int[]{64, 64}), new Tensor(new int[]{64, 64}));

        //Linear
        Tensor tensor16 = tf.matmul(tensor15, new Tensor(new int[]{num, dim}));
        Tensor tensor17 = tf.addx(tensor16, new Tensor("bias", new int[]{4, 1}));
        Tensor tensor18 = tf.sigmoidx(tensor17);

        //Add & Normal
        Tensor tensor19 = tf.addx(tensor15, tensor18);
        Tensor tensor10 = tf.layerNormal(tensor19, new Tensor(new int[]{64, 64}), new Tensor(new int[]{64, 64}));

        //MultiHeadAttention
        Tensor tensor21 = tf.multiHeadAttention(8, tensor10, new Tensor(new int[]{8, 3, num, dim}), new Tensor(new int[]{64, 64}), new Tensor(new int[]{64, 64}), new Tensor(new int[]{64, 64}));

        //Linear
        Tensor tensor22 = tf.matmul(tensor21, new Tensor(new int[]{num, dim}));
        Tensor tensor23 = tf.addx(tensor22, new Tensor("bias", new int[]{4, 1}));
        Tensor tensor24 = tf.sigmoidx(tensor23);

        //Add & Normal
        Tensor tensor25 = tf.addx(tensor21, tensor24);
        Tensor tensor26 = tf.layerNormal(tensor25, new Tensor(new int[]{64, 64}), new Tensor(new int[]{64, 64}));

        //Softmax & Loss
        Tensor label = new Tensor(new int[]{num, 1});
        Tensor softmax = tf.softmax(tensor26);
        Tensor crossx = tf.softmaxCrossx(label, softmax);

        TensorExecutor executor = new TensorExecutor(crossx);
        forEach(100000000, i -> {
            executor.run(null, null);
            if (i % 1000 == 0) {
                log.info("---------{}------------", i);
                Tensor loss = crossx.getOutput().one();
                log("输入：", null);
                log("标签：", null);
                log("输出：", softmax.data());
                log("误差：", loss.data());
            }
        });

        System.out.println(JSONObject.toJSONString(tensor18));
    }

    public void log(String name, Object obj) {
        log.info(name.concat(JSONObject.toJSONString(obj)));
    }
}