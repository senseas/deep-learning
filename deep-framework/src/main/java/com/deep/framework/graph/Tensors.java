package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

@Data
@Accessors(chain = true)
public class Tensors implements Serializable {

    public Tensors(Tensor tensor) {
        this.shape = tensor.getShape();
        this.size = Shape.size(shape);
        this.data = tensor.getData();
        this.grad = tensor.getGrad();
        this.nexts = next();
    }

    public Tensors(double[] data, double[] grad, int[] shape) {
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = data;
        this.grad = grad;
        this.nexts = next();
    }

    private Tensors(double[] data, double[] grad, int[] shape, int start) {
        this.start = start;
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = data;
        this.grad = grad;
        this.nexts = next();
    }

    public Tensors get(int... index) {
        return new Tensors(this.data, this.grad, getNext(index), start(index));
    }

    public double[] getData() {
        return streamData().toArray();
    }

    public double[] getGrad() {
        return streamGrad().toArray();
    }

    private DoubleStream streamData() {
        return IntStream.range(0, size()).mapToDouble(i -> data[start + i]);
    }

    private DoubleStream streamGrad() {
        return IntStream.range(0, size()).mapToDouble(i -> grad[start + i]);
    }

    private int start(int[] index) {
        int next = this.start, length = index.length - 1;
        for (int i = 0; i < length; i++) next += index[i] * nexts[i];
        return next + index[length] * nexts[length];
    }

    private int[] next() {
        int[] next = new int[shape.length];
        Arrays.fill(next, 1);
        for (int i = next.length - 1; 0 < i; i--) next[i - 1] = next[i] * shape[i];
        return next;
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

    public int shape(int i) {return shape[i];}

    public int size() {return size;}

    private int[] shape, nexts;
    private double[] data, grad;
    private int start = 0, size = 1;
}