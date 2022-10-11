package com.deep.framework.graph;

import com.deep.framework.framework.CudaContext;
import com.deep.framework.framework.TensorFlux;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

@Data
public class Tensor implements Serializable {

    public Tensor(double value) {
        this.name = "None";
        this.value = value;
        this.grad = 0d;
        this.gradre = true;
        this.output = new None(this);
    }

    public Tensor(double value, boolean gradre) {
        this.name = "None";
        this.value = value;
        this.grad = 0d;
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
        this.output = input;
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public <M> Tensor(M m) {
        this.name = "Function";
        this.function = m;
        this.output = TensorFlux.getOutput(function);
    }

    public <M> M compute() { return null; }

    public void gradient() { }

    public void forward() { }

    public void backward() { }

    public void reduce() { }

    public <M> M getOutput() { return (M) output; }

    public int shape(int i) {return shape[i];}

    public CudaContext getContext() {
        if (Objects.nonNull(context)) return context;
        return context = new CudaContext(this);
    }

    private String name = "Tensor::";
    protected int[] shape;
    private Tensor[] input;
    protected Object output, value, grad;
    protected transient Object function, reduce;
    private transient boolean gradre;
    private transient CudaContext context;
    private transient boolean out;
}
