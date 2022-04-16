package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.deep.framework.framework.TensorExecutor;
import com.deep.framework.framework.TensorFlow;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.DataLoader;
import com.deep.framework.lang.ModeLoader;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.function.Func;
import com.deep.framework.lang.util.ImageUtil;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class ResnetTest extends Shape {

    @Test
    public void DenseNetTest() {
        double[][][][] inputSet = DataLoader.getImageData();
        double[][][][] labelSet = DataLoader.getImageData();

        TensorFlow tf = new TensorFlow();
        Tensor input = new Tensor(new int[]{3, 140, 140});
        Tensor label = new Tensor(new int[]{3, 140, 140});

        Tensor tensor11 = tf.convx(new int[]{2, 2}, 0, new Tensor("weight", new int[]{64, 5, 5}), input);//64*134*134
        Tensor tensor12 = tf.relux(tensor11);//64*134*134
        Tensor tensor13 = tf.maxpoolx(3,new int[]{2, 2}, 0, tensor12);//64*68*68

        Tensor tensor21 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{64, 3, 3}), tensor13);//64*68*68
        Tensor tensor22 = tf.relux(tensor21);//64*68*68

        Tensor tensor31 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{64, 3, 3}), tensor22);//64*68*68
        Tensor tensor32 = tf.addx(tensor31, tensor13);//64*68*68
        Tensor tensor33 = tf.relux(tensor32);//64*68*68

        Tensor tensor41 = tf.convx(new int[]{2, 2}, 1, new Tensor("weight", new int[]{128, 3, 3}), tensor33);//128*34*34
        Tensor tensor42 = tf.relux(tensor41);//128*34*34

        Tensor tensor51 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{128, 3, 3}), tensor42);//128*34*34
        Tensor tensor52 = tf.convx(new int[]{2, 2}, 0, new Tensor("weight", new int[]{128, 1, 1}), tensor33);//128*34*34
        Tensor tensor53 = tf.addx(tensor51, tensor52);//128*34*34
        Tensor tensor54 = tf.relux(tensor53);//128*34*34

        Tensor tensor61 = tf.convx(new int[]{2, 2}, 1, new Tensor("weight", new int[]{256, 3, 3}), tensor54);//256*17*17
        Tensor tensor62 = tf.relux(tensor61);//256*17*17

        Tensor tensor71 = tf.convx(new int[]{1, 1}, 1, new Tensor("weight", new int[]{256, 3, 3}), tensor62);//256*17*17
        Tensor tensor72 = tf.convx(new int[]{2, 2}, 0, new Tensor("weight", new int[]{256, 1, 1}), tensor54);//256*17*17
        Tensor tensor73 = tf.addx(tensor72, tensor71);//256*17*17
        Tensor tensor74 = tf.relux(tensor73);//256*17*17

        Tensor squarex = tf.squarex(label, tensor74);

        TensorExecutor.rate = 0.03;
        TensorExecutor executor = new TensorExecutor(squarex, input, label);
        forEach(600, x -> {
            forEach(labelSet.length, i -> {
                log.info("---------{}:{}------------", x, i);
                Object inSet = inputSet[i], labSet = labelSet[i];
                executor.run(inSet, labSet);
                ModeLoader.save(executor, DataLoader.BASE_PATH.concat(i + "LetNet.obj"));
                Double[][][] data = Shape.reshape(tensor74.getOutput(), new Double[3][140][140], (Func<Tensor>) (Tensor a) -> (double) a.getValue());
                ImageUtil.rgb2Image(data, "D:/img/".concat(i + ".jpg"));
                log(squarex.getOutput());
            });
        });
    }

    public void log(Object obj) {
        log.info(JSONObject.toJSONString(obj));
    }
}
