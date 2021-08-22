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
        this.lengths = getLength();
    }

    private Tenser(T[] data, int[] shape, int start) {
        this.shape = shape;
        this.data = data;
        this.start = start;
        this.lengths = getLength();
    }

    public Tenser(int[] shape) {
        this.shape = shape;
        this.data = randomx(shape);
        this.start = 0;
        this.lengths = getLength();
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
        int start = start(index), end = end(index);
        for (int i = start - 1; i < end; i++) {
            this.data[i] = data[i - start + 1];
        }
    }

    public void set(T data, int... index) {
        int start = start(index);
        this.data[start] = data;
    }

    private int start(int[] index) {
        int next = this.start, length = index.length - 1;
        for (int i = 0; i < length; i++) {
            next += index[i] * lengths[i];
        }
        return next + index[length];
    }

    private int end(int[] index) {
        int next = this.start, length = index.length;
        for (int i = 0; i < length; i++) {
            next += index[i] * lengths[i];
        }
        return next;
    }

     int[] getLength() {
        int next = 1;
        int[] length = new int[shape.length - 1];
        for (int i = length.length; 0 < i; i--) {
            length[i - 1] = next *= shape[i];
        }
        return length;
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

}