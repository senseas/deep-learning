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
    int batch_size = 500, header_num = 8, dim = 512;
    int num = getWordDicSize();
    double scaler = 1 / Math.pow(512, 0.5);
    String data = "10月13日，@京东发言人发文称，我们关注到有谣言称“刘姓商人涉嫌违法被抓”，该谣言被别有用心的人刻意发布在京东相关新闻动态下，以混淆视听、操纵舆论。我们对此恶劣行径表示强烈愤慨，并已向公安机关报案。";
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

        Tensor tensor11 = tf.matmul(input, new Tensor(new int[]{num, dim}));//500*512
        Tensor tensor12 = tf.positionalEmbedding(tensor11.getShape(), inputx);//500*512
        Tensor tensor13 = tf.addx(tensor11, tensor12);//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor14 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor13);//500*512
        Tensor tensor15 = tf.linear(tensor14, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor16 = tf.addx(tensor14, tensor15);//500*512
        Tensor tensor17 = tf.layerNormal(tensor16, new Tensor(tensor16.getShape()), new Tensor(tensor16.getShape()));//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor21 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor17);//500*512
        Tensor tensor22 = tf.linear(tensor21, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor23 = tf.addx(tensor21, tensor22);//500*512
        Tensor tensor24 = tf.layerNormal(tensor23, new Tensor(tensor23.getShape()), new Tensor(tensor23.getShape()));//500*512

        //Linear
        Tensor tensor25 = tf.linear(tensor24, new Tensor(new int[]{dim, num}));//500*61224
        Tensor tensor26 = tf.linear(new Tensor(new int[]{1, batch_size}), tensor25);//1*61224

        //Softmax & Loss
        Tensor label = new Tensor(new int[]{1, num});
        Tensor softmax = tf.softmax(tensor26);
        Tensor crossx = tf.softmaxCrossx(label, softmax);

        TensorExecutor executor = new TensorExecutor(crossx, input, inputx, label);
        IntStream.range(2, 102).forEach(i -> {
            List<String> data = list.stream().limit(i).collect(Collectors.toList());
            double[] inxSet = pad(data);
            double[] inySet = getWordIndex(data);
            double[] labSet = oneHot(list.get(i));

            long start = System.currentTimeMillis();
            executor.run(inxSet, inySet, labSet);
            System.out.println(System.currentTimeMillis() - start);

            log.info("---------{}------------", i);
            log("输入：", String.join("", data));
            log("输出：", list.get(i));
            log("损失：", crossx.data());
        });
    }

    public double[] pad(List<String> data) {
        IntStream.range(data.size(), batch_size).forEach(i -> data.add("<pad>"));
        return getOneHotData(data);
    }

    public void log(String name, Object obj) {
        log.info(name.concat(JSONObject.toJSONString(obj)));
    }
}