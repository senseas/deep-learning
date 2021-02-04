package com.deep.framework.lang;

import lombok.Getter;

import java.util.Arrays;

@Getter
public class Tenser<T> {

    private final int[] shape;
    private final T[] data;

    public Tenser(int[] shape, T[] data) {
        this.data = data;
        this.shape = shape;
    }

    public <E> E get(Integer index) {
        if (shape.length == 1) return (E) data[index];
        int row = data.length / shape[0];
        int[] d = Arrays.copyOfRange(shape, 1, shape.length);
        T[] b = Arrays.copyOfRange(data, index * row, index * row + row);
        return (E) new Tenser(d, b);
    }

    public <E> E get(Integer... index) {
        Object a = this.get(index[0]);
        if (index.length == 1) return (E) a;
        index = Arrays.copyOfRange(index, 1, index.length);
        Tenser b = (Tenser) a;
        return (E) b.get(index);
    }

}