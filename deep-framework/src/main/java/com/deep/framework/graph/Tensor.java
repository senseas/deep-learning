package com.deep.framework.graph;

import com.deep.framework.core.TensorExecutor;
import com.deep.framework.cuda.CudaContext;
import com.deep.framework.lang.Tenser;
import lombok.Data;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

@Data
public class Tensor implements Serializable {
    static final double EX = 0.0000000001;

    public Tensor(double value) {
        this.name = "None";
        this.data = new double[]{value};
        this.grad = new double[]{0d};
        this.reduce = new boolean[]{false};
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.data = random(shape);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.shape = shape;
        this.data = random(shape);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
    }

    public Tensor(int[] shape, double value) {
        this.name = "None";
        this.shape = shape;
        this.data = values(shape, value);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
    }

    public Tensor(Tensor tensor, int idx) {
        this.idx = idx;
        this.data = tensor.getData();
        this.grad = tensor.getGrad();
        this.reduce = tensor.getReduce();
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public void forward() {
        if (Objects.nonNull(grad)) Arrays.fill(grad, 0d);
        if (Objects.nonNull(reduce)) Arrays.fill(reduce, false);
    }

    public void backward() { }

    public void reducer() {
        if (Objects.nonNull(reduce)) {
            forEach(getOutput(), (Tensor none) -> {
                if (!none.reduce()) {
                    none.reduce(true);
                    double valu = Math.abs(none.data()), grad = Math.abs(none.grad());
                    double rate = Math.min(valu / (grad + EX), grad / (valu + EX)) * TensorExecutor.rate;
                    double value = none.data() - rate * none.grad();
                    none.data(value);
                }
            });
        }
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        if (Objects.isNull(shape)) return output = new Tenser<>(new Tensor[]{this});
        return output = Tensors(this);
    }

    public double data() {
        return this.data[idx];
    }

    public void data(double value) {
        this.data[idx] = value;
    }

    public void grad(double grad) {
        this.grad[idx] += grad;
    }

    public double grad() {
        return this.grad[idx];
    }

    public boolean reduce() {
        return this.reduce[idx];
    }

    public void reduce(boolean reduce) {
        this.reduce[idx] = reduce;
    }

    public int shape(int i) {
        return shape[i];
    }

    public CudaContext getContext() {
        if (Objects.nonNull(context)) return context;
        return context = new CudaContext(this);
    }

    transient private int idx;
    protected int[] shape;
    protected double[] data, grad;
    transient protected boolean[] reduce;

    private String name = "Tensor::";
    private Tensor[] input;
    transient protected Tenser<Tensor> output, function;
    transient private CudaContext context;
}