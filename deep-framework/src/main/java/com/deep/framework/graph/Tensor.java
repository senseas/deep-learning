package com.deep.framework.graph;

import com.deep.framework.core.TensorExecutor;
import com.deep.framework.creater.ParamCreater;
import com.deep.framework.cuda.CudaContext;
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
        this.value = new double[]{value};
        this.grad = new double[]{0d};
        this.reduce = new boolean[]{false};
        this.gradre = true;
        this.output = new None(this);
    }

    public Tensor(double value, boolean gradre) {
        this.name = "None";
        this.value = new double[]{value};
        this.grad = new double[]{0d};
        this.reduce = new boolean[]{false};
        this.gradre = gradre;
        this.output = new None(this);
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.value = random(shape);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
        this.gradre = true;
        this.output = fillNones(this);
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.shape = shape;
        this.value = random(shape);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
        this.gradre = true;
        this.output = fillNones(this);
    }

    public Tensor(int[] shape, double value, boolean gradre) {
        this.name = "None";
        this.shape = shape;
        this.value = values(shape, value);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
        this.gradre = gradre;
        this.output = fillNones(this);
    }

    public Tensor(None input) {
        this.name = "None";
        this.gradre = input.isGradre();
        this.output = input;
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public void forward() {
        if (Objects.nonNull(value)) {
            Arrays.fill(reduce, false);
            Arrays.fill(grad, 0d);
        }
    }

    public void backward() { }

    public void reduce() {
        if (Objects.nonNull(reduce)) {
            forEach(getOutput(), (None none) -> {
                if (none.isGradre() && !none.isReduce()) {
                    none.setReduce(true);
                    double valu = Math.abs(none.getValue()), grad = Math.abs(none.getGrad());
                    double rate = Math.min(valu / (grad + EX), grad / (valu + EX)) * TensorExecutor.rate;
                    double value = none.getValue() - rate * none.getGrad();
                    none.setValue(value);
                }
            });
        }
    }

    public <M> M getOutput() { return (M) output; }

    public int shape(int i) {return shape[i];}

    public CudaContext getContext() {
        if (Objects.nonNull(context)) return context;
        return context = new CudaContext(this);
    }

    private String name = "Tensor::";
    private Tensor[] input;
    protected transient Object output, function;
    protected int[] shape;
    protected double[] value, grad;
    protected double valuex, gradx;
    protected transient boolean[] reduce;
    private transient boolean gradre;
    private transient CudaContext context;
    private transient ParamCreater core;
}