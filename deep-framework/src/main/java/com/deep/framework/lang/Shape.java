package com.deep.framework.lang;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorConst;
import com.deep.framework.lang.function.Func;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
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
        return (E) fill(a, shape(Tensor.class, a), o -> new TensorConst(0d));
    }

    public static <E> E zeroTensors(int[] a, int[] b) {
        int length = Arrays.stream(a).reduce((x, y) -> x * y).getAsInt();
        Tensor[] tensors = IntStream.range(0, length).mapToObj(c -> new Tensor(b, 0d, false)).toArray(Tensor[]::new);
        return (E) new Tenser(tensors, a);
    }

    public static <E> E zeroNones(Object o) {
        if (o instanceof Tenser) {
            int[] shape = shapes(o);
            return (E) new Tenser(fill(shape, 0d, false), shape);
        } else if (o instanceof int[]) {
            int[] shape = (int[]) o;
            return (E) new Tenser(fill(shape, 0d, false), shape);
        }
        return null;
    }

    public static <E> E fill(int[] x, double value, boolean isGrad) {
        int length = Arrays.stream(x).reduce((a, b) -> a * b).getAsInt();
        return (E) IntStream.range(0, length).mapToObj(a -> new None(value, isGrad)).toArray(None[]::new);
    }

    public static Object shape(Class clas, Object a) {
        int[] shape = shapes(a);
        int length = Arrays.stream(shape).reduce((x, y) -> x * y).getAsInt();
        Object data = Array.newInstance(clas, new int[]{length});
        return new Tenser((Object[]) data, shape);
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
        while (Objects.nonNull(arr) && arr instanceof Tenser) {
            list.add(Tensers.getLength(arr));
            arr = Tensers.get(arr, 0);
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    public static double[] linesValue(Object arr) {
        double[] list = new double[size(arr)];
        AtomicInteger index = new AtomicInteger();
        forEach(arr, (None a) -> {
            list[index.getAndIncrement()] = a.getValue();
        });
        return list;
    }

    public static double[] linesGrad(Object arr) {
        double[] list = new double[size(arr)];
        AtomicInteger index = new AtomicInteger();
        forEach(arr, (None a) -> {
            list[index.getAndIncrement()] = a.getGrad();
        });
        return list;
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


