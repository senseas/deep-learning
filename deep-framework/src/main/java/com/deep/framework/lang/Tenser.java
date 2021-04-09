package com.deep.framework.lang;

import lombok.Getter;

import java.util.Arrays;
import java.util.stream.IntStream;

import static com.deep.framework.lang.Shape.randomx;

@Getter
public class Tenser<T> {

    private final int[] shape;
    private final T[] data;
    private final int length, start, end;

    public Tenser(int[] shape, T[] data) {
        this.length = shape[0];
        this.shape = shape;
        this.data = data;
        this.start = 0;
        this.end = this.length - 1;
    }

    public Tenser(int[] shape, T[] data, int start, int end) {
        this.length = shape[0];
        this.shape = shape;
        this.data = data;
        this.start = start;
        this.end = end;
    }

    public Tenser(int[] shape) {
        this.length = shape[0];
        this.shape = shape;
        this.data = randomx(shape);
        this.start = 0;
        this.end = this.length - 1;
    }

    public <E> E get(Integer... index) {
        int start = this.start + IntStream.range(0, index.length).map(i -> reduce(shape, index[i], i + 1, shape.length)).sum();
        int end = start + reduce(shape, 1, index.length, shape.length);
        if (index.length == shape.length) return (E) data[end];
        int[] d = Arrays.copyOfRange(shape, index.length, shape.length);
        return (E) new Tenser(d, data, start, end);
    }

    public void set(T[] data, Integer... index) {
        int start = this.start + IntStream.range(0, index.length).map(i -> reduce(shape, index[i], i + 1, shape.length)).sum();
        int end = start + reduce(shape, 1, index.length, shape.length);
        if ((start + data.length) != end) throw new IndexOutOfBoundsException(String.valueOf(start + data.length));
        assert (start + data.length) == end : "out index";
        IntStream.range(0, data.length).forEach(i -> this.data[i + start] = data[i]);
    }

    private int reduce(int[] s, int i, int form, int to) {
        return Arrays.stream(s, form, to).reduce(i, (a, b) -> a * b);
    }

}