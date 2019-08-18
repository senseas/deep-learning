package com.deep.framework.graph;

import com.deep.framework.bean.Node;
import com.deep.framework.bean.None;
import lombok.Data;

@Data
public class Tenser<N> implements Node<N> {

    public Tenser(Double input) {
        this.name = "None";
        this.output = (N) new None(input);
        this.graph = new Graph();
    }

    public Tenser(int[] shape) {
        this.name = "None";
        this.output = Shape.random(shape);
        this.graph = new Graph();
    }

    public Tenser(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.output = Shape.random(this.name, shape);
        this.graph = new Graph();
    }

    public Tenser(None input) {
        this.name = input.getName();
        this.output = (N) input;
        this.graph = new Graph();
    }

    public Tenser(String name, Node... input) {
        this.name = this.name.concat(name);
        this.input = input;
        this.graph = new Graph();
    }

    public <M> Tenser(M[] m) {
        this.name = "Tenser";
        this.function = (N) m;
        this.graph = new Graph();
    }

    public <M> M compute() { return null; }

    public void gradient() { }

    public <M> M getInput(int i) {
        return Builder.build(this, i);
    }

    public N getFunction() {
        if (function != null) return function;
        Builder.function(this);
        return function;
    }

    public N getOutput() {
        if (output != null) return output;
        output = Shape.Nones(function);
        return output;
    }

    private String name = "Tenser::";
    private Node[] input;
    private transient Graph<Tenser> graph;
    private transient N function;
    private N output;
}
