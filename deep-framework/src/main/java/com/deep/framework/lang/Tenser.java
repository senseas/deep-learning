package com.deep.framework.lang;

import java.util.Arrays;
import java.util.stream.IntStream;

import static com.deep.framework.lang.Shape.randomx;
import static java.util.stream.IntStream.range;

public class Tenser<T> {

    private final T[] data;
    private final int[] shape;
    private final int length, start;

    public Tenser(T[] data, int[] shape) {
        this.length = shape[0];
        this.shape = shape;
        this.data = data;
        this.start = 0;
    }

    private Tenser(T[] data, int[] shape, int start) {
        this.length = shape[0];
        this.shape = shape;
        this.data = data;
        this.start = start;
    }

    public Tenser(int[] shape) {
        this.length = shape[0];
        this.shape = shape;
        this.data = randomx(shape);
        this.start = 0;
    }

    public <E> E get(int... index) {
        int start = getIndex(index);
        if (index.length == this.shape.length) return (E) this.data[start];
        return (E) new Tenser(this.data, getNext(index), start);
    }

    public void set(T[] data, int... index) {
        int start = getIndex(index);
        int end = start + reduce(getNext(index), 1, 0);
        range(start, end).forEach(i -> this.data[i] = data[i - start]);
    }

    private int getIndex(int[] index) {
        range(0, index.length).forEach(i -> {
            if (index[i] >= this.shape[i]) throw new IndexOutOfBoundsException(String.valueOf(index[i]));
        });
        return this.start + range(0, index.length).map(i -> reduce(this.shape, index[i], i + 1)).sum();
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

    private int reduce(int[] s, int i, int form) {
        return Arrays.stream(s, form, s.length).reduce(i, (a, b) -> a * b);
    }

    public static void main(String[] args) {
        Tenser<Double> tenser = new Tenser(new Double[800 * 200 * 300], new int[]{800, 200, 300});
        long s = System.currentTimeMillis();
        IntStream.range(0, 800).forEach(l -> {
            IntStream.range(0, 200).forEach(m -> {
                IntStream.range(0, 300).forEach(n -> {
                    tenser.get(l, m, n);
                });
            });
        });
        System.out.println((System.currentTimeMillis() - s) / 1000d);

        double[][][] data1 = new double[800][200][300];
        s = System.currentTimeMillis();
        IntStream.range(0, 800).forEach(l -> {
            IntStream.range(0, 200).forEach(m -> {
                IntStream.range(0, 300).forEach(n -> {
                    double v = data1[l][m][n];
                });
            });
        });
        System.out.println((System.currentTimeMillis() - s) / 1000d);

        double[] data2 = new double[800 * 200 * 300];
        s = System.currentTimeMillis();
        IntStream.range(0, 800).forEach(l -> {
            int x = l * 200 * 300;
            IntStream.range(0, 200).forEach(m -> {
                int y = m * 300;
                IntStream.range(0, 300).forEach(n -> {
                    double v = data2[x + y + n];
                });
            });
        });
        System.out.println((System.currentTimeMillis() - s) / 1000d);
    }

}