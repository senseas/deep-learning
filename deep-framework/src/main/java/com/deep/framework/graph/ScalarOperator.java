package com.deep.framework.graph;

import java.util.Arrays;
import java.util.stream.Stream;

public class ScalarOperator extends Tensor {

    public ScalarOperator(String name, Tensor... input) {
        super(name, input);
        this.value = new double[1];
        this.grad = new double[1];
        this.output = new None(this);
    }

    public double compute() { return 0; }

    public void gradient(double grad) { }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        value[0] = compute();
    }

    public void backward() {
        gradient(grad[0]);
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        for (Tensor o : getInput()) o.reduce();
    }

    public void clearOutput() {
        value[0] = 0;
        grad[0] = 0;
    }

    public void clearGrad() {
        grad[0] = 0;
    }

    public None getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Stream inputStream() {
        return Arrays.stream(getInput()).map(Tensor::getOutput);
    }

    public <M> M getOutput() {
        return (M) output;
    }

}