package com.deep.framework.lang;

import lombok.Getter;

import java.util.Arrays;

import static com.deep.framework.lang.Shape.randomx;

@Getter
public class Tenser<T> {

    private final int[] shape;
    private final T[] data;
    private final int length;

    public Tenser(int[] shape, T[] data) {
        this.length = shape[0];
        this.shape = shape;
        this.data = data;
    }

    public Tenser(int[] shape) {
        this.length = shape[0];
        this.shape = shape;
        this.data = randomx(shape);
    }

    public <E> E get(int index) {
        if (shape.length == 1) return (E) data[index];
        int row = data.length / length;
        int[] d = Arrays.copyOfRange(shape, 1, shape.length);
        T[] b = Arrays.copyOfRange(data, index * row, index * row + row);
        return (E) new Tenser(d, b);
    }

    public <E> E get(int... index) {
        Object a = this.get(index[0]);
        if (index.length == 1) return (E) a;
        int[] indexs = Arrays.copyOfRange(index, 1, index.length);
        Tenser b = (Tenser) a;
        return (E) b.get(indexs);
    }

}