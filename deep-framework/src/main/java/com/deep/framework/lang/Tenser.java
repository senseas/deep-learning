package com.deep.framework.lang;

import java.util.Arrays;

import static com.deep.framework.lang.Shape.randomx;

public class Tenser<T> {

    private final T[] data;
    private final int[] shape, lengths;
    private final int start;

    public Tenser(T[] data, int[] shape) {
        this.shape = shape;
        this.data = data;
        this.start = 0;
        this.lengths = getLength(shape);
    }

    private Tenser(T[] data, int[] shape, int start) {
        this.shape = shape;
        this.data = data;
        this.start = start;
        this.lengths = getLength(shape);
    }

    public Tenser(int[] shape) {
        this.shape = shape;
        this.data = randomx(shape);
        this.start = 0;
        this.lengths = getLength(shape);
    }

    public <E> E get(int... index) {
        int start = start(index);
        if (index.length == this.shape.length) {
            return (E) this.data[start];
        } else {
            return (E) new Tenser(this.data, getNext(index), start);
        }
    }

    public void set(T[] data, int... index) {
        int start = start(index);
        if (index.length == this.shape.length) {
            this.data[start] = data[0];
        } else {
            int end = end(index);
            for (int i = start; i <= end; i++) {
                this.data[i - 1] = data[i - start];
            }
        }
    }

    public void set(T data, int... index) {
        int start = start(index);
        this.data[start] = data;
    }

    private int start(int[] index) {
        int next = this.start, length = index.length;
        for (int i = 0; i < length - 1; i++) {
            next += index[i] * lengths[i];
        }
        return next += index[length - 1];
    }

    private int end(int[] index) {
        int next = this.start, length = index.length;
        for (int i = 0; i < length; i++) {
            next += index[i] * lengths[i];
        }
        return next;
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

}