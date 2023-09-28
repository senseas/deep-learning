package com.deep.framework.graph;

import com.deep.framework.core.TensorExecutor;
import com.deep.framework.cuda.CudaContext;
import com.deep.framework.lang.Tenser;
import lombok.Data;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.IntStream;

import static com.deep.framework.lang.Shape.*;

@Data
public class Tensor implements Serializable {
    static final double EX = 0.0000000001;

    public Tensor(double value) {
        this.name = "None";
        this.data = new double[]{value};
        this.grads = new double[]{0d};
        this.reduces = new boolean[]{false};
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.data = random(shape);
        this.grads = zeros(shape);
        this.reduces = booleans(shape);
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.shape = shape;
        this.data = random(shape);
        this.grads = zeros(shape);
        this.reduces = booleans(shape);
        this.output = fillNones(this);
    }

    public Tensor(int[] shape, double value) {
        this.name = "None";
        this.shape = shape;
        this.data = values(shape, value);
        this.grads = zeros(shape);
        this.reduces = booleans(shape);
        this.output = fillNones(this);
    }

    public Tensor(Tensor tensor, int idx) {
        this.idx = idx;
        this.data = tensor.getData();
        this.grads = tensor.getGrads();
        this.reduces = tensor.getReduces();
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public void forward() {
        if (Objects.nonNull(data)) {
            Arrays.fill(reduces, false);
            Arrays.fill(grads, 0d);
        }
    }

    public void backward() { }

    public void reduce() {
        if (Objects.nonNull(reduces)) {
            forEach(getOutput(), (Tensor none) -> {
                if (!none.getReduce()) {
                    none.setReduce(true);
                    double valu = Math.abs(none.getValue()), grad = Math.abs(none.getGrad());
                    double rate = Math.min(valu / (grad + EX), grad / (valu + EX)) * TensorExecutor.rate;
                    double value = none.getValue() - rate * none.getGrad();
                    none.setValue(value);
                }
            });
        }
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        output = new Tenser<>(Tensor.class, shape);
        IntStream.range(0, output.size()).forEach(i -> output.set(new Tensor(this, i), i));
        return output;
    }

    public double getValue() {
        return data[idx];
    }

    public void setValue(double value) {
        this.data[idx] = value;
    }

    public void setGrad(double grad) {
        this.grads[idx] += grad;
    }

    public double getGrad() {
        return grads[idx];
    }

    public boolean getReduce() {
        return reduces[idx];
    }

    public void setReduce(boolean reduce) {
        this.reduces[idx] = reduce;
    }

    public int shape(int i) {return shape[i];}

    public CudaContext getContext() {
        if (Objects.nonNull(context)) return context;
        return context = new CudaContext(this);
    }

    private String name = "Tensor::";
    private Tensor[] input;
    protected transient Tenser<Tensor> output, function;
    protected double[] data, grads;
    protected transient boolean[] reduces;
    protected int[] shape;
    private transient int idx;
    private transient CudaContext context;
}