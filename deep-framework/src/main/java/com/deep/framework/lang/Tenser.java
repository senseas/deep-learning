package com.deep.framework.lang;

import java.util.Arrays;
import java.util.stream.IntStream;

import static com.deep.framework.lang.Shape.randomx;

public class Tenser<T> {

    private final T[] data;
    private final int[] shape, lengths;
    private final int start;

    public Tenser(T[] data, int[] shape) {
        this.shape = shape;
        this.data = data;
        this.lengths = getLength(shape);
        this.start = 0;
    }

    private Tenser(T[] data, int[] shape, int start) {
        this.shape = shape;
        this.data = data;
        this.lengths = getLength(shape);
        this.start = start;
    }

    public Tenser(int[] shape) {
        this.shape = shape;
        this.data = randomx(shape);
        this.lengths = getLength(shape);
        this.start = 0;
    }

    public <E> E get(int... index) {
        int start = getIndex(index);
        if (index.length == this.shape.length) {
            return (E) this.data[start];
        } else {
            return (E) new Tenser(this.data, getNext(index), start);
        }
    }

    public void set(T[] data, int... index) {
        int start = getIndex(index);
        if (index.length == this.shape.length) {
            this.data[start] = data[0];
        } else {
            int end = index.length * lengths[0];
            for (int i = start; i < end; i++) {
                this.data[i] = data[i - start];
            }
        }
    }

    private int getIndex(int[] index) {
        int next = this.start, length = index.length;
        for (int i = 0; i < length - 1; i++) {
            next += index[i] * lengths[i];
        }
        return next += index[length - 1];
    }

    public static int[] getLength(int[] shape) {
        int[] length = new int[shape.length - 1];
        for (int i = length.length, next = 1; 0 < i; i--) {
            next *= shape[i];
            length[i - 1] = next;
        }
        return length;
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
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