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
        if(status) return;
        for (Tensor o : getInput()) o.forward();
        clearOutput();

        Tensor tensor = getFunction().data(0);
        tensor.forward();
        data[0] = tensor.data();
        status = true;
    }

    public void backward() {
        if (states) return;
        for (Tensor o : getInput()) o.setStatus(o.states).setStates(true);

        Tensor tensor = getFunction().data(0);
        tensor.grad(grad[0]);
        tensor.backward();
        clearGrad();

        for (Tensor o : getInput()) o.setStates(o.status).setStatus(false).backward();
    }

    public void reducer() {
        if(states) return;
        getFunction().forEach(Tensor::reducer);
        for (Tensor o : getInput()) o.reducer();
        states = true;
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = new Tenser<>(compute());
    }

    public void clearOutput() {
        states = false;
        data[0] = 0;
        grad[0] = 0;
    }

    public void clearGrad() {
        status = false;
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