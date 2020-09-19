package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;

@Data
public class Tensor<N> implements Serializable {

    public Tensor(Double input) {
        this.name = "None";
        this.output = (N) new None(input);
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.output = Shape.random(shape);
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.output = Shape.random(this.name, shape);
    }

    public Tensor(None input) {
        this.name = input.getName();
        this.output = (N) input;
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public <M> Tensor(M[] m) {
        this.name = "Tensor";
        this.function = (N) m;
    }

    public <M> M compute() { return null; }

    public void gradient() { }

    public <M> M getInput(int i) {
        return Builder.getInput(this, i);
    }

    public N getFunction() {
        if (function != null) return function;
        Builder.function(this);
        return function;
    }

    private String name = "Tensor::";
    private Tensor[] input;
    private transient N function;
    private N output;
}
