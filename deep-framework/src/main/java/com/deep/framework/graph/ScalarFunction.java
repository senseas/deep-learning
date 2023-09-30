package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

import static com.deep.framework.lang.ForEach.forEach;

public class ScalarFunction extends Tensor{

    public ScalarFunction(String name, Tensor... input) {
        super(name, input);
        this.data = new double[1];
        this.grads = new double[1];
    }

    public Tensor compute() { return null; }

    public void gradient() { }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();

        Tensor tensor = getFunction().one();
        tensor.forward();
        data[0] = tensor.getValue();
    }

    public void backward() {
        Tensor tensor = getFunction().one();
        tensor.setGrad(grads[0]);
        tensor.backward();

        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        forEach(getFunction(), Tensor::reduce);
        for (Tensor o : getInput()) o.reduce();
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = new Tenser<>(new Tensor[]{compute()});
    }

    public void clearOutput() {
        data[0] = 0;
        grads[0] = 0;
    }

    public void clearGrad() {
        grads[0] = 0;
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = new Tenser<>(new Tensor[]{new Tensor(this, 0)});
    }

}