package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.functions.Operator;

import java.util.Objects;

import static com.deep.framework.lang.ForEach.forEach;

public class ScalarFunction extends Tensor implements Operator {

    public ScalarFunction(String name, Tensor... input) {
        super(name, input);
        this.value = new double[1];
        this.grad = new double[1];
        this.output = new None(this);
    }

    public Tensor compute() { return null; }

    public void gradient() { }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();

        Tensor function = (Tensor) getFunction();
        function.forward();
        value[0] = function.getValue()[0];
    }

    public void backward() {
        Tensor function = (Tensor) getFunction();
        function.getGrad()[0] = grad[0];
        function.backward();

        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        forEach(getFunction(), Tensor::reduce);
        for (Tensor o : getInput()) o.reduce();
    }

    public Object getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = compute();
    }

    public void clearOutput() {
        value[0] = 0;
        grad[0] = 0;
    }

    public void clearGrad() {
        grad[0] = 0;
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        return TensorFlux.getTensor(input.getOutput());
    }

    public <M> M getOutput() { return (M) output; }

}