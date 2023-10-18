package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.deep.framework.lang.util.Sequence.*;

@Slf4j
public class TransformerTest {
    int batch_size = 500;
    int num = getWordDicSize(), dim = 512;
    String data = "10月13日，@京东发言人 发文称，我们关注到有谣言称“刘姓商人涉嫌违法被抓”，该谣言被别有用心的人刻意发布在京东相关新闻动态下，以混淆视听、操纵舆论。我们对此恶劣行径表示强烈愤慨，并已向公安机关报案。";
    List<String> list = getWordIndexList2(data);

    /**
     * gpt decoder
     * 模型初步搭建
     */
    @Test
    public void transformerTest() {
        TensorFlow tf = new TensorFlow();
        //Embedding
        Tensor input = new Tensor(new int[]{batch_size, num});//500*16224
        Tensor inputx = new Tensor(new int[]{batch_size});

        Tensor tensor12 = tf.matmul(input, new Tensor(new int[]{num, dim}));//500*512
        Tensor tensor13 = tf.positionalEmbedding(tensor12.getShape(), inputx);//500*512
        Tensor tensor14 = tf.addx(tensor12, tensor13);//500*512

        //MultiHeadAttention
        Tensor tensor15 = tf.multiHeadAttention(64, tensor14, new Tensor(new int[]{8, 3, dim, dim}), new Tensor(new int[]{dim * 8, dim}), new Tensor(new int[]{batch_size, dim}), new Tensor(new int[]{batch_size, dim}));//500*512

        //Linear
        Tensor tensor16 = tf.matmul(tensor15, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor17 = tf.addx(tensor16, new Tensor("bias", tensor16.getShape()));//500*512
        Tensor tensor18 = tf.relux(tensor17);//500*512

        //Add & Normal
        Tensor tensor19 = tf.addx(tensor15, tensor18);//500*512
        Tensor tensor10 = tf.layerNormal(tensor19, new Tensor(tensor19.getShape()), new Tensor(tensor19.getShape()));//500*512

        //MultiHeadAttention
        Tensor tensor21 = tf.multiHeadAttention(8, tensor10, new Tensor(new int[]{8, 3, dim, dim}), new Tensor(new int[]{dim * 8, dim}), new Tensor(new int[]{batch_size, dim}), new Tensor(new int[]{batch_size, dim}));//500*512

        //Linear
        Tensor tensor22 = tf.matmul(tensor21, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor23 = tf.addx(tensor22, new Tensor("bias", tensor22.getShape()));//500*512
        Tensor tensor24 = tf.relux(tensor23);//500*512

        //Linear
        Tensor tensor25 = tf.addx(tensor21, tensor24);//500*512
        Tensor tensor26 = tf.layerNormal(tensor25, new Tensor(tensor25.getShape()), new Tensor(tensor25.getShape()));//500*512

        //Linear
        Tensor tensor27 = tf.matmul(tensor26, new Tensor(new int[]{dim, num}));//500*61224
        Tensor tensor28 = tf.addx(tensor27, new Tensor("bias", tensor27.getShape()));//500*61224
        Tensor tensor29 = tf.relux(tensor28);//500*61224

        //Linear
        Tensor tensor31 = tf.matmul(new Tensor(new int[]{1, batch_size}), tensor29);//1*61224
        Tensor tensor32 = tf.addx(tensor31, new Tensor("bias", tensor31.getShape()));//1*16224
        Tensor tensor33 = tf.relux(tensor32);//1*16224

        //Softmax & Loss
        Tensor label = new Tensor(new int[]{1, num});
        Tensor softmax = tf.softmax(tensor33);
        Tensor crossx = tf.softmaxCrossx(label, softmax);

        TensorExecutor executor = new TensorExecutor(crossx, input, inputx, label);
        IntStream.range(2, 102).forEach(i -> {
            List<String> data = list.stream().limit(i).collect(Collectors.toList());
            double[] inSet = pad(data);

            double[] labSet = oneHot(list.get(i + 1));
            double[] wordIndex = getWordIndex(data);
            executor.run(inSet, wordIndex, labSet);
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

    public double[] pad(List<String> data) {
        IntStream.range(data.size(), batch_size).forEach(i -> data.add("<pad>"));
        return getOneHotData(data);
    }

    public void log(String name, Object obj) {
        log.info(name.concat(JSONObject.toJSONString(obj)));
    }
}