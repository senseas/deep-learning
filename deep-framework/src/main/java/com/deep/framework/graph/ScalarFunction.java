package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

public class ScalarFunction extends Tensor {

    public ScalarFunction(String name, Tensor... input) {
        super(name, input);
        this.data = new double[1];
        this.grad = new double[1];
    }

    public Tensor compute() { return null; }

    public void gradient() { }

    public void forward() {
        clearOutput();
        getFunction().forEach(a -> data[0] = a.data());
    }

    public void backward() {
        getFunction().forEach(a -> a.grad(grad[0]));
        clearGrad();
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = new Tenser<>(compute());
    }

    public void clearOutput() {
        data[0] = 0;
        grad[0] = 0;
    }

    public void clearGrad() {
        grad[0] = 0;
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = new Tenser<>(new Tensor(this, 0));
    }

}