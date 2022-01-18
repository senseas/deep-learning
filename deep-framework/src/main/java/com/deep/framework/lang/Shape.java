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

    public static <E> E random(int[] shape) {
        RandomDataGenerator random = new RandomDataGenerator();
        return (E) IntStream.range(0, size(shape)).parallel().mapToDouble(i -> random.nextGaussian(0, 0.1)).toArray();
    }

    public static <E> E values(int[] shape, double value) {
        return (E) IntStream.range(0, size(shape)).parallel().mapToDouble(i -> value).toArray();
    }

    public static <E> E fillNones(Tensor tensor) {
        None[] nones = IntStream.range(0, size(tensor.getShape())).parallel().mapToObj(i -> new None(tensor, i)).toArray(None[]::new);
        return (E) new Tenser(nones, tensor.getShape());
    }

    public static double[] zeros(int[] shape) {
        return new double[size(shape)];
    }

    public static boolean[] booleans(int[] shape) {
        return new boolean[size(shape)];
    }

    public static <E> E zeroTensors(Object a) {
        return (E) fill(shape(Tensor.class, a), o -> new TensorConst(0d));
    }

    public static <E> E zeroTensors(int[] a, int[] b) {
        return (E) fill(shape(Tensor.class, a), o -> new TensorConst(b, 0d));
    }

    public static <E> E zeroNones(Object a) {
        return (E) fill(shape(None.class, a), o -> new None(0d, false));
    }

    public static Object shape(Class clas, Object o) {
        if (o instanceof Tenser) {
            int[] shape = shapes(o);
            return new Tenser(clas, shape);
        } else if (o instanceof int[]) {
            int[] shape = (int[]) o;
            return new Tenser(clas, shape);
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
        arrayEach(B, (b, i) -> Array.set(b, i, fill.apply(link.poll())));
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