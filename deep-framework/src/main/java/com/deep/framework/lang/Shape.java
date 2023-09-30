package com.deep.framework.lang;

import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorConst;
import com.deep.framework.lang.function.Func;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.*;
import java.util.stream.IntStream;

public class Shape extends ForEach {

    public static double[] random(int[] shape) {
        RandomDataGenerator random = new RandomDataGenerator();
        return IntStream.range(0, size(shape)).mapToDouble(i -> random.nextGaussian(0, 0.1)).toArray();
    }

    public static double[] values(int[] shape, double value) {
        return IntStream.range(0, size(shape)).mapToDouble(i -> value).toArray();
    }

    public static Tenser<Tensor> Tensors(Tensor tensor) {
        Tensor[] tensors = IntStream.range(0, size(tensor.getShape())).mapToObj(i -> new Tensor(tensor, i)).toArray(Tensor[]::new);
        return new Tenser<>(tensors, tensor.getShape());
    }

    public static Tenser<Tensor> TensorConsts(Tensor tensor) {
        Tensor[] tensors = IntStream.range(0, size(tensor.getShape())).mapToObj(i -> new TensorConst(tensor, i)).toArray(Tensor[]::new);
        return new Tenser<>(tensors, tensor.getShape());
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
        return (E) fill(shape(Tensor.class, a), o -> new TensorConst(0d, b));
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
        ferEach(A, B, (a, b, i) -> b[i] = fill.apply(a));
        return (M) B;
    }

    public static int[] shapes(Object arr) {
        List<Integer> list = new ArrayList();
        if (arr instanceof Tenser) {
            while (Objects.nonNull(arr) && arr instanceof Tenser) {
                list.add(Tensers.getLength(arr));
                arr = Tensers.get(arr, 0);
            }
        } else if (arr instanceof int[]) {
            return (int[]) arr;
        } else if (arr.getClass().isArray()) {
            while (Objects.nonNull(arr) && arr.getClass().isArray()) {
                list.add(Array.getLength(arr));
                arr = Array.get(arr, 0);
            }
        }
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    public static int[] shape(int... shape) {
        return shape;
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