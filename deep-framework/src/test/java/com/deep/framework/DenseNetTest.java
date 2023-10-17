package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.core.TensorExecutor;
import com.deep.framework.core.TensorFlow;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.DataLoader;
import com.deep.framework.lang.ModeLoader;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.function.Func;
import com.deep.framework.lang.util.ImageUtil;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static com.deep.framework.lang.ForEach.forEach;

@Slf4j
public class DenseNetTest{

    @Test
    public void DenseNetTest3() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{10, 3, 3}), input);//16*140*140
        Tensor tensor12 = tf.relux(tensor11);//16*140*140
        Tensor tensor13 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{10, 3, 3}), tensor12);//16*140*140
        Tensor tensor14 = tf.relux(tensor13);//16*140*140
        Tensor tensor15 = tf.maxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor14);//16*70*70

        Tensor tensor21 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{20, 3, 3}), tensor15);//20*70*70
        Tensor tensor22 = tf.relux(tensor21);//20*70*70
        Tensor tensor23 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{20, 3, 3}), tensor22);//20*70*70
        Tensor tensor24 = tf.relux(tensor23);//20*70*70
        Tensor tensor25 = tf.maxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor24);//20*35*35

        Tensor tensor31 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{40, 3, 3}), tensor25);//32*35*35
        Tensor tensor32 = tf.relux(tensor31);//32*35*35

        Tensor tensor41 = tf.demaxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor32);//32*70*70
        Tensor tensor42 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{20, 3, 3}), tensor41);//20*70*70
        Tensor tensor43 = tf.relux(tensor42);//20*70*70
        Tensor tensor44 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{20, 3, 3}), tensor43);//20*70*70
        Tensor tensor45 = tf.relux(tensor44);//20*70*70

        Tensor tensor51 = tf.demaxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor45);//20*140*140
        Tensor tensor52 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{10, 3, 3}), tensor51);//16*140*140
        Tensor tensor53 = tf.relux(tensor52);//16*140*140
        Tensor tensor54 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{10, 3, 3}), tensor53);//16*140*140
        Tensor tensor55 = tf.relux(tensor54);//16*69*69

        Tensor tensor61 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{8, 3, 3}), tensor55);//3*140*140
        Tensor tensor62 = tf.relux(tensor61);//10*140*140
        Tensor tensor63 = tf.convx(new int[]{1, 1}, new int[]{1, 1}, new Tensor("weight", new int[]{3, 3, 3}), tensor62);//3*140*140
        //Tensor tensor64 = tf.relux(tensor63);//10*140*140
        Tensor squarex = tf.squarex(label, tensor63);

        TensorExecutor.rate = 0.003;
        TensorExecutor executor = new TensorExecutor(squarex, input, label);
        forEach(600, labelSet.length, (x, i) -> {
            log.info("epoch = {{},{}}", x, i);
            Object inSet = inputSet[i], labSet = labelSet[i];
            executor.run(inSet, labSet);
            if (x != 0 && x % 2 == 0) ModeLoader.save(executor, i + "LetNet.obj");
            img(tensor63, i);
            log(squarex.getData());
        });
    }

    @Test
    public void DenseNetTest5() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{16, 5, 5}), input);//16*140*140
        Tensor tensor12 = tf.relux(tensor11);//16*140*140
        Tensor tensor13 = tf.maxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor12);//16*140*140

        Tensor tensor21 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{32, 5, 5}), tensor13);//32*70*70
        Tensor tensor22 = tf.relux(tensor21);//32*70*70
        Tensor tensor23 = tf.maxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor22);//32*35*35

        Tensor tensor31 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{64, 5, 5}), tensor23);//64*35*35
        Tensor tensor32 = tf.relux(tensor31);//64*35*35
        Tensor tensor33 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{64, 5, 5}), tensor32);//64*35*35
        Tensor tensor34 = tf.relux(tensor33);//64*35*35

        Tensor tensor41 = tf.demaxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor34);//64*70*70
        Tensor tensor42 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{32, 5, 5}), tensor41);//32*70*70
        Tensor tensor43 = tf.relux(tensor42);//32*70*70

        Tensor tensor51 = tf.demaxpoolx(new int[]{2, 2}, new int[]{2, 2}, new int[]{0, 0}, tensor43);//32*140*140
        Tensor tensor52 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{16, 5, 5}), tensor51);//16*140*140
        Tensor tensor53 = tf.relux(tensor52);//16*140*140

        Tensor tensor61 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{8, 5, 5}), tensor53);//3*140*140
        Tensor tensor62 = tf.relux(tensor61);//16*140*140
        Tensor tensor63 = tf.convx(new int[]{1, 1}, new int[]{2, 2}, new Tensor("weight", new int[]{3, 5, 5}), tensor62);//3*140*140
        Tensor squarex = tf.squarex(label, tensor63);

        TensorExecutor.rate = 0.003;
        TensorExecutor executor = new TensorExecutor(squarex, input, label);
        forEach(600, labelSet.length, (x, i) -> {
            log.info("epoch = {{},{}}", x, i);
            Object inSet = inputSet[i], labSet = labelSet[i];
            executor.run(inSet, labSet);
            ModeLoader.save(executor, i + "LetNet.obj");
            img(tensor63, i);
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
            img(output, i);
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
        img(output, 0);
        log(squarex.getOutput());
    }

    public void img(Tensor tensor, int i) {
        Func<Tensor> fun = Tensor::data;
        Double[][][] data = Shape.reshape(tensor.getOutput(), new Double[3][140][140], fun);
        ImageUtil.rgb2Image(data, DataLoader.IMG_PATH.concat(i + ".jpg"));
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }

}
