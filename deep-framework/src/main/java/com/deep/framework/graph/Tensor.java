package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import lombok.Data;

import java.io.Serializable;

@Data
public class Tensor<N> implements Serializable {

    public Tensor(double input) {
        this.name = "None";
        this.output = (N) new None(input);
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.output = Shape.random(shape);
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.output = Shape.random(shape);
    }

    public Tensor(None input) {
        this.name = "None";
        this.output = (N) input;
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public <M> Tensor(M[] m) {
        this.name = "Function";
        this.function = (N) m;
    }

    public <M> M compute() { return null; }

    public void gradient() { }

    public void forward() { }

    public void backward() { }

    public void reduce() { }

    private String name = "Tensor::";
    private Tensor[] input;
    protected transient N function;
    protected N output;
}
