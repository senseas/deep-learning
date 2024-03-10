package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.ModeLoader;
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
    List<List<String>> list = getMedicalTokenList();

    @Test
    public void TransformerTest() {
        TensorFlow tf = new TensorFlow();
        //Embedding
        Tensor input = new Tensor(new int[]{batch_size, num});//500*16224
        Tensor inputx = new Tensor(new int[]{batch_size});

        Tensor tensor11 = tf.matmul(input, new Tensor(new int[]{num, dim}));//500*512
        Tensor tensor12 = tf.positionalEmbedding(tensor11.getShape(), inputx);//500*512
        Tensor tensor13 = tf.addx(tensor11, tensor12);//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor21 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor13);//500*512
        Tensor tensor22 = tf.linear(tensor21, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor23 = tf.addx(tensor21, tensor22);//500*512
        Tensor tensor24 = tf.layerNormal(tensor23, new Tensor(tensor23.getShape()), new Tensor(tensor23.getShape()));//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor31 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor24);//500*512
        Tensor tensor32 = tf.linear(tensor31, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor33 = tf.addx(tensor31, tensor32);//500*512
        Tensor tensor34 = tf.layerNormal(tensor33, new Tensor(tensor33.getShape()), new Tensor(tensor33.getShape()));//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor41 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor34);//500*512
        Tensor tensor42 = tf.linear(tensor41, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor43 = tf.addx(tensor41, tensor42);//500*512
        Tensor tensor44 = tf.layerNormal(tensor43, new Tensor(tensor43.getShape()), new Tensor(tensor43.getShape()));//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor51 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor44);//500*512
        Tensor tensor52 = tf.linear(tensor51, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor53 = tf.addx(tensor51, tensor52);//500*512
        Tensor tensor54 = tf.layerNormal(tensor53, new Tensor(tensor53.getShape()), new Tensor(tensor53.getShape()));//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor61 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor54);//500*512
        Tensor tensor62 = tf.linear(tensor61, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor63 = tf.addx(tensor61, tensor62);//500*512
        Tensor tensor64 = tf.layerNormal(tensor63, new Tensor(tensor63.getShape()), new Tensor(tensor63.getShape()));//500*512

        //MultiHeadAttention & Linear Add & Normal
        Tensor tensor71 = tf.multiHeadAttention(batch_size, dim, header_num, scaler, tensor64);//500*512
        Tensor tensor72 = tf.linear(tensor71, new Tensor(new int[]{dim, dim}));//500*512
        Tensor tensor73 = tf.addx(tensor71, tensor72);//500*512
        Tensor tensor74 = tf.layerNormal(tensor73, new Tensor(tensor73.getShape()), new Tensor(tensor73.getShape()));//500*512

        //Linear
        Tensor tensor55 = tf.linear(tensor74, new Tensor(new int[]{dim, num}));//500*61224
        Tensor tensor56 = tf.linear(new Tensor(new int[]{1, batch_size}), tensor55);//1*61224

        //Softmax & Loss
        Tensor label = new Tensor(new int[]{1, num});
        Tensor softmax = tf.softmax(tensor56);
        Tensor crossx = tf.softmaxCrossx(label, softmax);

        TensorExecutor executor = new TensorExecutor(crossx, input, inputx, label);
        list.forEach(words -> {
            IntStream.range(1, words.size()).forEach(i -> {
                List<String> data = getInput(words, i);
                double[] inxSet = getFlatTokenOneHotList(data);
                double[] inySet = getTokenIndex(data);
                double[] labSet = oneHot(words.get(i));

                long start = System.currentTimeMillis();
                executor.run(inxSet, inySet, labSet);
                System.out.println(System.currentTimeMillis() - start);

                log.info("---------{}------------", i);
                log("输入：", String.join("", data));
                log("输出：", words.get(i));
                log("损失：", crossx.data());
            });
            ModeLoader.save(executor, "gpt.ml");
        });
    }

    @Test
    public void TrainTest() {
        TensorExecutor executor = ModeLoader.load("gpt.ml");
        Tensor crossx = executor.getTensor();
        list.forEach(words -> {
            IntStream.range(1, words.size()).forEach(i -> {
                List<String> data = getInput(words, i);
                double[] inxSet = getFlatTokenOneHotList(data);
                double[] inySet = getTokenIndex(data);
                double[] labSet = oneHot(words.get(i));

                long start = System.currentTimeMillis();
                executor.run(inxSet, inySet, labSet);
                System.out.println(System.currentTimeMillis() - start);

                log.info("---------{}------------", i);
                log("输入：", String.join("", data));
                log("输出：", words.get(i));
                log("损失：", crossx.data());
            });
        });
    }

    public List<String> getInput(List<String> words, int index) {
        List<String> data = words.stream().limit(index).collect(Collectors.toList());
        IntStream.range(data.size(), batch_size).forEach(i -> data.add("<pad>"));
        return data;
    }

    public void log(String name, Object obj) {
        log.info(name.concat(JSONObject.toJSONString(obj)));
    }
}