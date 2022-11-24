package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class TensorFunction extends Tensor {

    public TensorFunction(String name, Tensor... input) {
        super(name, input);
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        if (BeanUtil.isOperation(input)) return (M) input;
        if (BeanUtil.isFunction(input)) return (M) input.getFunction();
        return TensorFlux.getTensor(input.getOutput());
    }

    public Object getOutput() {
        if (Objects.nonNull(output)) return output;
        if (Objects.nonNull(getFunction())) output = TensorFlux.getOutput(function);
        return output;
    }

    public Object getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = compute();
    }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        TensorFlux.forward(this);
    }

    public void backward() {
        TensorFlux.backward(this);
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        TensorFlux.reduce(this);
        for (Tensor o : getInput()) o.reduce();
    }

}
