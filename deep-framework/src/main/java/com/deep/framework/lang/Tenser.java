package com.deep.framework.lang;

import com.deep.framework.graph.None;

import java.util.Arrays;

import static com.deep.framework.lang.Shape.randomx;
import static java.util.stream.IntStream.range;

public class Tenser<T> {

    private final T[] data;
    private final int[] shape;
    private final int length, start;

    public Tenser(T[] data, int[] shape) {
        this.length = shape[0];
        this.data = data;
        this.shape = shape;
        this.start = 0;
    }

    private Tenser(T[] data, int[] shape, int start) {
        this.length = shape[0];
        this.data = data;
        this.shape = shape;
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
        Tenser<None> tenser = new Tenser(new int[]{3, 2, 3});
        None o = tenser.get(0, 0, 2);
        Tenser<None>x =tenser.get(0, 0);
        None o1 = x.get(2);
        tenser.set(new None[]{new None(0d), new None(1d), new None(2d)}, 1, 0, 2);
        System.out.println(o);
    }

}