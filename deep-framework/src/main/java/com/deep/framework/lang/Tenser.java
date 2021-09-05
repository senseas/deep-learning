package com.deep.framework.lang;

import java.lang.reflect.Array;
import java.util.Arrays;

import static com.deep.framework.lang.Shape.*;

public class Tenser<T> {

    public final Object data;
    public final int[] shape;

    public Tenser(T data, int[] shape) {
        this.shape = shape;
        this.data = data;
    }

    public Tenser(int[] shape) {
        this.shape = shape;
        this.data = random(shape);
    }

    public Tenser(Class clas, int[] shape) {
        this.shape = shape;
        this.data = Array.newInstance(clas, shape);
    }

    public <E> E get(int... index) {
        Object data = this.data;
        for (int i : index) data = Array.get(data, i);
        if (index.length == this.shape.length) {
            return (E) data;
        } else {
            return (E) new Tenser(data, getNext(index));
        }
    }

    public void set(Object data, int... index) {
        Object d = this.data;
        for (int i = 0; i < index.length - 1; i++) {
            d = Array.get(d, i);
        }
        Array.set(d, index[index.length - 1], data);
    }

    public int shape(int i) {return shape[i];}

    public int getLength() {return Array.getLength(data);}

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

}