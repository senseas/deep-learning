package com.deep.gradient.graph;

import com.deep.gradient.operation.base.None;

import java.lang.reflect.Array;
import java.util.stream.IntStream;

public class Shape {

    public static void forEach(int a, Each1 e) {
        IntStream.range(0, a).forEach(i -> e.apply(i));
    }

    public static void forEach(int a, int b, Each2 e) {
        IntStream.range(0, a).forEach(i -> IntStream.range(0, b).forEach(l -> e.apply(i, l)));
    }

    public static void forEach(int a, int b, int c, Each3 e) {
        IntStream.range(0, a).forEach(i -> IntStream.range(0, b).forEach(l -> IntStream.range(0, c).forEach(m -> e.apply(i, l, m))));
    }

    public static Object createTenser(int... x) {
        return Array.newInstance(None.class, x);
    }

    @FunctionalInterface
    public interface Each1 {
        void apply(int l);
    }

    @FunctionalInterface
    public interface Each2 {
        void apply(int l, int i);
    }

    @FunctionalInterface
    public interface Each3 {
        void apply(int l, int i, int m);
    }
}


