package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenser;
import lombok.Data;

import java.io.Serializable;

@Data
public class Tensor implements Serializable {

    public Tensor(double value) {
        this.name = "None";
        this.output = new None(value);
    }

    public Tensor(double value, boolean isGrad) {
        this.name = "None";
        this.output = new None(value, isGrad);
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.output = new Tenser(shape);
    }

    public Tensor(int[] shape, double value, boolean isGrad) {
        this.name = "None";
        this.output = Shape.fill(shape, value, isGrad);
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.output = new Tenser(shape);
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
    }

    public <M> M compute() { return null; }

    public void gradient() { }

    public void forward() { }

    public void backward() { }

    public void reduce() { }

    public <M> M getOutput() { return (M) output; }

    private String name = "Tensor::";
    private Tensor[] input;
    protected transient Object function;
    protected Object output;
}
