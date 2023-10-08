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
    public final int[] shape;
    private final int[] nexts;
    private final int start;

    public Tenser(T[] data, int[] shape) {
        this.shape = shape;
        this.data = data;
        this.start = 0;
        this.nexts = next();
    }

    private Tenser(T[] data, int[] shape, int start) {
        this.shape = shape;
        this.data = data;
        this.start = start;
        this.nexts = next();
    }

    public Tenser(T data) {
        this.shape = new int[]{1};
        this.data = (T[]) new Object[]{data};
        this.start = 0;
        this.nexts = next();
    }

    public Tenser(Class clas, int[] shape) {
        this.shape = shape;
        this.data = (T[]) Array.newInstance(clas, size());
        this.start = 0;
        this.nexts = next();
    }

    public <E> E get(int... index) {
        int start = start(index);
        if (index.length == this.shape.length) {
            return (E) this.data[start];
        } else {
            return (E) new Tenser(this.data, getNext(index), start);
        }
    }

    public <E> E data(int index) {
        return (E) this.data[start + index];
    }

    public void set(T[] data, int... index) {
        int start = start(index), end = end(start, index);
        for (int i = start; i < end; i++) {
            this.data[i] = data[i - start];
        }
    }

    public void set(T data, int... index) {
        int start = start(index);
        this.data[start] = data;
    }

    public void set(T data, int index) {
        this.data[start + index] = data;
    }

    private int start(int[] index) {
        int next = this.start, length = index.length - 1;
        for (int i = 0; i < length; i++) {
            next += index[i] * nexts[i];
        }
        return next + index[length] * nexts[length];
    }

    private int end(int start, int[] index) {
        int length = index.length - 1;
        return start + index[length] * nexts[length];
    }

    private int[] next() {
        int[] next = new int[shape.length];
        Arrays.fill(next, 1);
        for (int i = next.length - 1; 0 < i; i--) {
            next[i - 1] = next[i] * shape[i];
        }
        return next;
    }

    public int shape(int i) {
        return shape[i];
    }

    public int getLength() {
        return shape[0];
    }

    public T tensor() {
        return data[start];
    }

    public int size() {
        return Shape.size(shape);
    }

    public void forEach(For<T> func) {
        IntStream.range(0, size()).forEach(i -> func.apply(data[start + i], i));
    }

    public void forEach(Func1<T> func) {
        IntStream.range(0, size()).forEach(i -> func.apply(data[start + i]));
    }

    public Stream<T> stream() {
        return IntStream.range(0, size()).mapToObj(i -> data[start + i]);
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

}