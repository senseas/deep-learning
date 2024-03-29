package com.deep.framework.lang;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorConst;
import com.deep.framework.lang.function.Func;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.IntStream;

public class Shape extends ForEach {

    public static <E> E random(int[] x) {
        RandomDataGenerator random = new RandomDataGenerator();
        return (E) fill(Array.newInstance(None.class, x), o -> new None(random.nextGaussian(0, 0.1)));
    }

    public static <E> E randomx(int... x) {
        RandomDataGenerator random = new RandomDataGenerator();
        int length = Arrays.stream(x).reduce((a, b) -> a * b).getAsInt();
        return (E) IntStream.range(0, length).mapToObj(a -> new None(random.nextGaussian(0, 0.1))).toArray(None[]::new);
    }

    public static <E> E zeroTensors(Object a) {
        return (E) fill(shape(Tensor.class, a), o -> new TensorConst(0d));
    }

    public static <E> E zeroTensors(Object a, int[] b) {
        return (E) fill(a, o -> new TensorConst(b, 0d));
    }

    public static <E> E zeroNones(Object a) {
        return (E) fill(shape(None.class, a), o -> new None(0d, false));
    }

    public static <E> E fill(int[] x, double value, boolean isGrad) {
        return (E) fill(shape(None.class, x), o -> new None(value, isGrad));
    }

    public static Object shape(Class clas, Object a) {
        if (a instanceof int[]) {
            int[] shape = (int[]) a;
            return Array.newInstance(clas, shape);
        } else {
            int[] shape = shapes(a);
            return Array.newInstance(clas, shape);
        }
    }

    public static <M> M reshape(Object A, Object B) {
        Queue link = new LinkedList();
        forEach(A, a -> link.add(a));
        forEach(B, (b, i) -> b[i] = link.poll());
        return (M) B;
    }

    public static <M> M reshape(Object A, Object B, Func fill) {
        Queue link = new LinkedList();
        forEach(A, a -> link.add(a));
        forEach(B, (b, i) -> b[i] = fill.apply(link.poll()));
        return (M) B;
    }

    public static int[] shapes(Object arr) {
        List<Integer> list = new ArrayList();
        while (Objects.nonNull(arr) && arr.getClass().isArray()) {
            list.add(Array.getLength(arr));
            arr = Array.get(arr, 0);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    public static int size(Object arr) {
        int size = 1;
        while (Objects.nonNull(arr) && arr.getClass().isArray()) {
            size *= Array.getLength(arr);
            arr = Array.get(arr, 0);
        }
        return size;
    }

}