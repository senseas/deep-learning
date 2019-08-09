package com.deep.gradient.graph;

import com.deep.gradient.operation.Node;
import com.deep.gradient.operation.base.None;
import lombok.Data;

@Data
public class Tenser<N> implements Node {

    public Tenser(Double input) {
        this.name = "None";
        this.output = new None(input);
    }

    public Tenser(int[] shape) {
        this.name = "None";
        this.output = Shape.createTenser(shape);
    }

    public Tenser(String name, Node... input) {
        this.name = this.name.concat("::").concat(name);
        this.input = input;
        this.graph = new Graph();
        setGraph(input);
    }

    public N compute() {
        return null;
    }

    public void gradient() {
    }

    public void setGraph(Node... input) {

    }

    private Node[] input;
    private Graph graph;
    private Object output;
    private String name = "Tenser";
}
