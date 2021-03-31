package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.Executor;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFlow;
import com.deep.framework.lang.DataLoader;
import com.deep.framework.lang.ModeLoader;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.function.Fill;
import com.deep.framework.lang.util.ImageUtil;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class DenseNetTest extends Shape {

    @Test
    public void DenseNetTest5() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), input);//10*136*136
        Tensor tensor12 = tf.relux(tensor11);//10*136*136
        Tensor tensor13 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor12);//10*68*68

        Tensor tensor21 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor13);//16*64*64
        Tensor tensor22 = tf.relux(tensor21);//16*64*64
        Tensor tensor23 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor22);//16*32*32

        Tensor tensor31 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{32, 3, 3}), tensor23);//32*28*28
        Tensor tensor32 = tf.relux(tensor31);//32*28*28
        Tensor tensor33 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor32);//32*14*14

        Tensor tensor41 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor33);//32*28*28
        Tensor tensor42 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor41);//16*32*32
        Tensor tensor43 = tf.relux(tensor42);//32*28*28

        Tensor tensor51 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor43);//16*64*64
        Tensor tensor52 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), tensor51);//10*68*68
        Tensor tensor53 = tf.relux(tensor52);//16*64*64

        Tensor tensor61 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor53);//10*136*136
        Tensor tensor62 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{3, 3, 3}), tensor61);//3*140*140
        Tensor tensor63 = tf.relux(tensor62);//10*136*136
        Tensor squarex = tf.squarex(label, tensor63);

        Executor.rate = 0.03;
        Executor executor = new Executor(squarex, input, label);
        forEach(600, x -> {
            forEach(labelSet.length, i -> {
                log.info("---------{}:{}------------", x, i);
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                Double[][][] data = Shape.reshape(tensor63.getOutput(), new Double[3][140][140], (Fill<None>) (None a) -> (double) a.getValue());
                ImageUtil.rgb2Image(data, "D:/img/".concat(i + ".jpg"));
                log(squarex.getOutput());
            });
        });
    }

    @Test
    public void DenseNetTest5Relux() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), input);//10*136*136
        Tensor tensor12 = tf.relux(tensor11);//10*136*136
        Tensor tensor13 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor12);//10*68*68

        Tensor tensor21 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor13);//16*64*64
        Tensor tensor22 = tf.relux(tensor21);//16*64*64
        Tensor tensor23 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor22);//16*32*32

        Tensor tensor31 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{32, 3, 3}), tensor23);//32*28*28
        Tensor tensor32 = tf.relux(tensor31);//32*28*28
        Tensor tensor33 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor32);//32*14*14

        Tensor tensor41 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor33);//32*28*28
        Tensor tensor42 = tf.relux(tensor41);//32*28*28
        Tensor tensor43 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor42);//16*32*32

        Tensor tensor51 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor43);//16*64*64
        Tensor tensor52 = tf.relux(tensor51);//16*64*64
        Tensor tensor53 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), tensor52);//10*68*68

        Tensor tensor61 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor53);//10*136*136
        Tensor tensor62 = tf.relux(tensor61);//10*136*136
        Tensor tensor63 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{3, 3, 3}), tensor62);//3*140*140
        Tensor squarex = tf.squarex(label, tensor63);

        Executor.rate = 0.03;
        Executor executor = new Executor(squarex, input, label);
        forEach(600, x -> {
            forEach(labelSet.length, i -> {
                log.info("---------{}:{}------------", x, i);
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                Double[][][] data = Shape.reshape(tensor63.getOutput(), new Double[3][140][140], (Fill<None>) (None a) -> (double) a.getValue());
                ImageUtil.rgb2Image(data, "D:/img/".concat(i + ".jpg"));
                log(squarex.getOutput());
            });
        });
    }

    @Test
    public void DenseNetTest3() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), input);//10*138*138
        Tensor tensor12 = tf.relux(tensor11);//10*138*138
        Tensor tensor13 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor12);//10*69*69

        Tensor tensor21 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor13);//16*67*67
        Tensor tensor22 = tf.relux(tensor21);//16*67*67
        Tensor tensor23 = tf.maxpoolx(3, new int[]{2, 2}, 0, tensor22);//16*33*33

        Tensor tensor31 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{32, 3, 3}), tensor23);//32*31*31
        Tensor tensor32 = tf.relux(tensor31);//32*31*31
        Tensor tensor33 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor32);//32*15*15

        Tensor tensor41 = tf.demaxpoolx(3, new int[]{2, 2}, 0, tensor33);//32*31*31
        Tensor tensor42 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor41);//16*33*33
        Tensor tensor43 = tf.relux(tensor42);//16*33*33

        Tensor tensor51 = tf.demaxpoolx(3, new int[]{2, 2}, 0, tensor43);//16*67*67
        Tensor tensor52 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), tensor51);//10*69*69
        Tensor tensor53 = tf.relux(tensor52);//16*69*69

        Tensor tensor61 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor53);//10*138*138
        Tensor tensor62 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{3, 3, 3}), tensor61);//3*140*140
        Tensor tensor63 = tf.relux(tensor62);//10*140*140
        Tensor squarex = tf.squarex(label, tensor63);

        Executor.rate = 0.03;
        Executor executor = new Executor(squarex, input, label);
        forEach(600, x -> {
            forEach(labelSet.length, i -> {
                log.info("---------{}:{}------------", x, i);
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                Double[][][] data = Shape.reshape(tensor63.getOutput(), new Double[3][140][140], (Fill<None>) (None a) -> (double) a.getValue());
                ImageUtil.rgb2Image(data, "D:/img/".concat(i + ".jpg"));
                log(squarex.getOutput());
            });
        });
    }

    @Test
    public void DenseNetTest3Relux() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), input);//10*138*138
        Tensor tensor12 = tf.relux(tensor11);//10*138*138
        Tensor tensor13 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor12);//10*69*69

        Tensor tensor21 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor13);//16*67*67
        Tensor tensor22 = tf.relux(tensor21);//16*67*67
        Tensor tensor23 = tf.maxpoolx(3, new int[]{2, 2}, 0, tensor22);//16*33*33

        Tensor tensor31 = tf.convx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{32, 3, 3}), tensor23);//32*31*31
        Tensor tensor32 = tf.relux(tensor31);//32*31*31
        Tensor tensor33 = tf.maxpoolx(2, new int[]{2, 2}, 0, tensor32);//32*15*15

        Tensor tensor41 = tf.demaxpoolx(3, new int[]{2, 2}, 0, tensor33);//32*31*31
        Tensor tensor42 = tf.relux(tensor41);//16*33*33
        Tensor tensor43 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{16, 3, 3}), tensor42);//16*33*33

        Tensor tensor51 = tf.demaxpoolx(3, new int[]{2, 2}, 0, tensor43);//16*67*67
        Tensor tensor52 = tf.relux(tensor51);//16*69*69
        Tensor tensor53 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{10, 3, 3}), tensor52);//10*69*69

        Tensor tensor61 = tf.demaxpoolx(2, new int[]{2, 2}, 0, tensor53);//10*138*138
        Tensor tensor62 = tf.relux(tensor61);//10*140*140
        Tensor tensor63 = tf.deconvx(new int[]{1, 1}, 0, new Tensor("weight", new int[]{3, 3, 3}), tensor62);//3*140*140
        Tensor squarex = tf.squarex(label, tensor63);

        Executor.rate = 0.03;
        Executor executor = new Executor(squarex, input, label);
        forEach(600, x -> {
            forEach(labelSet.length, i -> {
                log.info("---------{}:{}------------", x, i);
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                Double[][][] data = Shape.reshape(tensor63.getOutput(), new Double[3][140][140], (Fill<None>) (None a) -> (double) a.getValue());
                ImageUtil.rgb2Image(data, "D:/img/".concat(i + ".jpg"));
                log(squarex.getOutput());
            });
        });
    }

    @Test
    public void TrainTest() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        Executor executor = ModeLoader.load(DataLoader.BASE_PATH.concat("LetNet.obj"));
        Executor.rate = 0.3;
        Tensor squarex = executor.getTensor();
        Tensor tensor63 = squarex.getInput()[1];
        forEach(600, x -> {
            forEach(labelSet.length, i -> {
                log.info("---------{}:{}------------", x, i);
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                Double[][][] data = Shape.reshape(tensor63.getOutput(), new Double[3][140][140], (Fill<None>) (None a) -> (double) a.getValue());
                ImageUtil.rgb2Image(data, "D:/img/".concat(i + ".jpg"));
                log(squarex.getOutput());
            });
        });
    }

    @Test
    public void EvalTest() {
        double[][][] inputSet = ImageUtil.image2RGB(DataLoader.BASE_PATH.concat("d.jpg"));
        double[][][] labelSet = ImageUtil.image2RGB(DataLoader.BASE_PATH.concat("d.jpg"));

        Executor executor = ModeLoader.load(DataLoader.BASE_PATH.concat("LetNet.obj"));
        Tensor squarex = executor.getTensor();
        Tensor tensor63 = squarex.getInput()[1];
        Object inSet = inputSet, labSet = labelSet;
        executor.forward(inSet, labSet);
        Double[][][] data = Shape.reshape(tensor63.getOutput(), new Double[3][140][140], (Fill<None>) (None a) -> (double) a.getValue());
        ImageUtil.rgb2Image(data, "D:/img/x.jpg");
        log(squarex.getOutput());
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
