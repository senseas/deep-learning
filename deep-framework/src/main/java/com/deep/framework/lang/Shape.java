package com.deep.framework.lang;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.util.BeanUtil;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.stream.IntStream;

public class Shape extends ForEach {

    static RandomDataGenerator random = new RandomDataGenerator();

    public static <E> E random(int... x) {
        return (E) fill(Array.newInstance(None.class, x), o -> new None(random.nextGaussian(0, 0.1)));
    }

    public static <E> E randomx(int... x) {
        int length = Arrays.stream(x).reduce((a, b) -> a * b).getAsInt();
        return (E) IntStream.range(0, length).mapToObj(a -> new None(random.nextGaussian(0, 0.1))).toArray(None[]::new);
    }

    public static <E> E zeros(Object a) {
        return (E) fill(a, o -> new Tensor(0d));
    }

    public static <E> E zeros(Object a, int b[]) {
        return (E) fill(a, o -> new Tensor(b));
    }

    public static Object shape(Class clas, Object a) {
        return Array.newInstance(clas, shapes(a));
    }

    public static <E> int[] shapes(E a, int... list) {
        if (BeanUtil.isTensor(a)) {
            int length = Array.getLength(list);
            list = Arrays.copyOf(list, length + 1);
            Array.set(list, length, Array.getLength(a));
            return shapes(Array.get(a, 0), list);
        }
        return list;
    }

    public static <M> M reshape(Object A, Object B) {
        Queue link = new LinkedList();
        forEach(A, a -> link.add(a));
        forEach(B, (b, i) -> b[i] = link.poll());
        return (M) B;
    }

}


