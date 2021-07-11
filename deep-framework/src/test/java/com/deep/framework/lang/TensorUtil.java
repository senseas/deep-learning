package com.deep.framework.lang;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class TensorUtil {

    public static void zeros(Tensor o, int... shape) {
        o.shape = shape;
        o.output = new float[Arrays.stream(shape).reduce((a, b) -> a * b).getAsInt()];
    }

    public static float[] random(int... shape) {
        Random r = new Random();
        int length = Arrays.stream(shape).reduce((a, b) -> a * b).getAsInt();
        float[] x = new float[length];
        IntStream.range(0, length).forEach(i -> x[i] = (float) (r.nextGaussian() * Math.sqrt(10)));
        return x;
    }

}
