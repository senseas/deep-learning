package com.deep.framework.lang;

import com.deep.framework.lang.function.For;
import com.deep.framework.lang.function.Func1;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class Tenser<T> implements Serializable {

    public final T[] data;
    public final int[] shape, nexts;
    private final int offset, size;

    public Tenser(T[] data, int[] shape) {
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = data;
        this.offset = 0;
        this.nexts = next();
    }

    private Tenser(T[] data, int[] shape, int offset) {
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = data;
        this.offset = offset;
        this.nexts = next();
    }

    public Tenser(T data) {
        this.shape = new int[]{1};
        this.size = Shape.size(shape);
        this.data = (T[]) new Object[]{data};
        this.offset = 0;
        this.nexts = next();
    }

    public Tenser(Class clas, int[] shape) {
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = (T[]) Array.newInstance(clas, size());
        this.offset = 0;
        this.nexts = next();
    }

    public <E> E get(int... index) {
        int start = offset(index);
        if (index.length == this.shape.length) {
            return (E) this.data[start];
        } else {
            return (E) new Tenser(this.data, getNext(index), start);
        }
    }

    public T data(int index) {
        return this.data[offset + index];
    }

    public void set(T[] data, int... index) {
        int start = offset(index), end = end(start, index);
        for (int i = start; i < end; i++) {
            this.data[i] = data[i - start];
        }
    }

    public void set(T data, int... index) {
        int start = offset(index);
        this.data[start] = data;
    }

    public void set(T data, int index) {
        this.data[offset + index] = data;
    }

    private int offset(int[] index) {
        int next = this.offset;
        for (int i = 0; i < index.length; i++) next += index[i] * nexts[i];
        return next;
    }

    private int end(int start, int[] index) {
        int length = index.length - 1;
        return start + index[length] * nexts[length];
    }

    private int[] next() {
        int[] next = new int[shape.length];
        next[next.length - 1] = 1;
        for (int i = next.length - 1; 0 < i; i--) next[i - 1] = next[i] * shape[i];
        return next;
    }

    public int shape(int i) {
        return shape[i];
    }

    public int getLength() {
        return shape[0];
    }

    public int size() {
        return size;
    }

    public void forEach(For<T> func) {
        for (int i = 0; i < size(); i++) {
            func.apply(data[offset + i], i);
        }
    }

    public void forEach(Func1<T> func) {
        for (int i = 0; i < size(); i++) {
            func.apply(data[offset + i]);
        }
    }

    public Stream<T> stream() {
        return IntStream.range(0, size()).mapToObj(i -> data[offset + i]);
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

}