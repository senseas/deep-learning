package com.deep.framework.lang;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorConst;
import com.deep.framework.lang.function.Func;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Shape extends ForEach {

    public static <E> E random(int[] shape) {
        RandomDataGenerator random = new RandomDataGenerator();
        return (E) fill(Array.newInstance(None.class, shape), o -> new None(random.nextGaussian(0, 0.1)));
    }

    public static <E> E zeroTensors(Object a) {
        return (E) fill(shape(Tensor.class, a), o -> new TensorConst(0d));
    }

    public static <E> E zeroTensors(int[] a, int[] b) {
        return (E) fill(shape(Tensor.class, a), o -> new TensorConst(b, 0d));
    }

    public static <E> E randomNones(int... shape) {
        RandomDataGenerator random = new RandomDataGenerator();
        return (E) fill(shape(None.class, shape), o -> new None(random.nextGaussian(0, 0.1)));
    }

    public static <E> E zeroNones(Object a) {
        return (E) fill(shape(None.class, a), o -> new None(0d, false));
    }

    public static <E> E fillNones(int[] shape, double value, boolean isGrad) {
        return (E) fill(shape(None.class, shape), o -> new None(value, isGrad));
    }

    public static Object shape(Class clas, Object o) {
        if (o instanceof Tenser) {
            int[] shape = shapes(o);
            int length = size(shape);
            Object data = Array.newInstance(clas, new int[]{length});
            return new Tenser((Object[]) data, shape);
        } else if (o instanceof int[]) {
            int[] shape = (int[]) o;
            int length = size(shape);
            Object data = Array.newInstance(clas, new int[]{length});
            return new Tenser((Object[]) data, shape);
        }
        return null;
    }

    public static <M> M reshape(Object A, Object B) {
        Queue link = new LinkedList();
        forEach(A, a -> link.add(a));
        forEach(B, (b, i) -> b.set(link.poll(), i));
        return (M) B;
    }

    public static <M> M reshape(Object A, Object B, Func fill) {
        Queue link = new LinkedList();
        forEach(A, a -> link.add(a));
        forEach(B, (b, i) -> b.set(fill.apply(link.poll()), i));
        return (M) B;
    }

    public static int[] shapes(Object arr) {
        List<Integer> list = new ArrayList();
        if (arr instanceof Tenser) {
            while (Objects.nonNull(arr) && arr instanceof Tenser) {
                list.add(Tensers.getLength(arr));
                arr = Tensers.get(arr, 0);
            }
        } else if (arr.getClass().isArray()) {
            while (Objects.nonNull(arr) && arr.getClass().isArray()) {
                list.add(Array.getLength(arr));
                arr = Array.get(arr, 0);
            }
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

    public static Class getArrayDeepClass(Object arr) {
        while (Objects.nonNull(arr) && arr.getClass().isArray()) arr = Array.get(arr, 0);
        return arr.getClass();
    }

    public static Class getTenserDeepClass(Tenser o) {
        return o.data.getClass().getComponentType();
    }

    public static int size(int[] arr) {
        int size = 1;
        for (int a : arr) {
            size *= a;
        }
        return size;
    }
}