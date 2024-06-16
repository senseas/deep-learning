package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

public class ScalarFunction extends Tensor {

    public ScalarFunction(String name, Tensor... input) {
        super(name, null, input);
    }

    public Tensor compute() { return null; }

    public void gradient() { }

    public void forward() {
        for (Tensor o : getInput()) o.setRefer(this).forward();

        create();
        clearOutput();
        getFunction().forEach(Tensor::forward);
        getFunction().forEach(a -> data[0] = a.data());
    }

    public void backward() {
        getFunction().forEach(a -> a.grad(grad[0]));
        getFunction().forEach(Tensor::backward);
        clearGrad();

        for (Tensor o : getInput()) o.setRefer(this).backward();
    }

    public void reducer() {
        getFunction().forEach(Tensor::reducer);
        for (Tensor o : getInput()) o.setRefer(this).reducer();
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = new Tenser<>(compute());
    }

    private void clearOutput() {
        data[0] = 0;
        grad[0] = 0;
    }

    private void clearGrad() {
        grad[0] = 0;
    }

    private void create() {
        if (Objects.nonNull(data)) return;
        this.data = new double[1];
        this.grad = new double[1];
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = new Tenser<>(this);
    }

}