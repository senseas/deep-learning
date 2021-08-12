package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.TensorExecutor;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.framework.TensorFlow;
import com.deep.framework.lang.DataLoader;
import com.deep.framework.lang.ModeLoader;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.function.Func;
import com.deep.framework.lang.util.ImageUtil;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class DenseNetTest extends Shape {

    @Test
    public void DenseNetTest3() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{10, 3, 3}), input);//16*140*140
        Tensor tensor12 = tf.relux(tensor11);//16*140*140
        Tensor tensor13 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{10, 3, 3}), tensor12);//16*140*140
        Tensor tensor14 = tf.relux(tensor13);//16*140*140

        Tensor tensor21 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{10, 3, 3}), tensor14);//20*70*70
        Tensor tensor22 = tf.relux(tensor21);//20*70*70
        Tensor tensor23 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{10, 3, 3}), tensor22);//20*70*70
        Tensor tensor24 = tf.relux(tensor23);//20*70*70

        Tensor tensor42 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{10, 3, 3}), tensor24);//20*70*70
        Tensor tensor43 = tf.relux(tensor42);//20*70*70
        Tensor tensor44 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{3, 3, 3}), tensor43);//20*70*70

        Tensor squarex = tf.squarex(label, tensor44);

        TensorExecutor.rate = 0.003;
        TensorExecutor executor = new TensorExecutor(squarex, input, label);
        forEach(60000, x -> {
            int i = (int) (Math.random() * labelSet.length);
            Object inSet = inputSet[i], labSet = labelSet[i];
            executor.forward(inSet, labSet);
            executor.backward();
            log(tensor11.getInput()[0].getOutput());
            log(tensor13.getInput()[0].getOutput());
            log(tensor21.getInput()[0].getOutput());
            log(tensor23.getInput()[0].getOutput());
            log(tensor42.getInput()[0].getOutput());
            log(tensor44.getInput()[0].getOutput());
            log(squarex.getOutput());
            log.info("epoch = {{},{}}", x, i);
            img(tensor44, String.valueOf(System.currentTimeMillis()));
            executor.reduce();
            if (x != 0 && x % 200 == 0) ModeLoader.save(executor, i + "LetNet.obj");
        });
    }

    @Test
    public void DenseNetTest5() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{16, 5, 5}), input);//16*140*140
        Tensor tensor12 = tf.relux(tensor11);//16*140*140
        Tensor tensor13 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor12);//16*140*140

        Tensor tensor21 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{32, 5, 5}), tensor13);//32*70*70
        Tensor tensor22 = tf.relux(tensor21);//32*70*70
        Tensor tensor23 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor22);//32*35*35

        Tensor tensor31 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{64, 5, 5}), tensor23);//64*35*35
        Tensor tensor32 = tf.relux(tensor31);//64*35*35
        Tensor tensor33 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{64, 5, 5}), tensor32);//64*35*35
        Tensor tensor34 = tf.relux(tensor33);//64*35*35

        Tensor tensor41 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor34);//64*70*70
        Tensor tensor42 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{32, 5, 5}), tensor41);//32*70*70
        Tensor tensor43 = tf.relux(tensor42);//32*70*70

        Tensor tensor51 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor43);//32*140*140
        Tensor tensor52 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{16, 5, 5}), tensor51);//16*140*140
        Tensor tensor53 = tf.relux(tensor52);//16*140*140

        Tensor tensor61 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{8, 5, 5}), tensor53);//3*140*140
        Tensor tensor62 = tf.relux(tensor61);//16*140*140
        Tensor tensor63 = tf.convx(new int[]{1, 1}, 2, new Tensor("weight", new int[]{3, 5, 5}), tensor62);//3*140*140
        Tensor squarex = tf.squarex(label, tensor63);

        TensorExecutor.rate = 0.003;
        TensorExecutor executor = new TensorExecutor(squarex, input, label);
        forEach(600, labelSet.length, (x, i) -> {
            log.info("epoch = {{},{}}", x, i);
            Object inSet = inputSet[i], labSet = labelSet[i];
            executor.run(inSet, labSet);
            ModeLoader.save(executor, i + "LetNet.obj");
            img(tensor63, String.valueOf(i));
            log(squarex.getOutput());
        });
    }

    @Test
    public void TrainTest() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorExecutor executor = ModeLoader.load("8LetNet.obj");
        TensorExecutor.rate = 0.003;
        Tensor squarex = executor.getTensor();
        Tensor output = squarex.getInput()[1];
        forEach(600, labelSet.length, (x, i) -> {
            log.info("epoch = {{},{}}", x, i);
            Object inSet = inputSet[i], labSet = labelSet[i];
            executor.run(inSet, labSet);
            ModeLoader.save(executor, i + "LetNet.obj");
            img(output, String.valueOf(i));
            log(squarex.getOutput());
        });
    }

    @Test
    public void EvalTest() {
        double[][][] inputSet = ImageUtil.image2RGB("d-140.jpg");
        double[][][] labelSet = ImageUtil.image2RGB("d-140.jpg");

        TensorExecutor executor = ModeLoader.load("LetNet.obj");
        Tensor squarex = executor.getTensor();
        Tensor output = squarex.getInput()[1];
        Object inSet = inputSet, labSet = labelSet;
        executor.forward(inSet, labSet);
        img(output, String.valueOf(0));
        log(squarex.getOutput());
    }

    public static void img(Tensor tensor, String i) {
        Func<None> fun = None::getValue;
        Double[][][] data = Shape.reshape(tensor.getOutput(), new Double[3][140][140], fun);
        ImageUtil.rgb2Image(data, DataLoader.IMG_PATH.concat(i + ".jpg"));
    }

    public static void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }

}
