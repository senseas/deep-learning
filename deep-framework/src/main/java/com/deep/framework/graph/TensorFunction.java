package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class TensorFunction extends Tensor {

    public TensorFunction(String name, Tensor... input) {
        super(name, input);
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        if (BeanUtil.isFunction(input)) return (M) input.getFunction();
        return TensorFlux.getTensor(input.getOutput());
    }

    public <M> M getOutput() {
        if (Objects.nonNull(output)) return (M) output;
        if (Objects.nonNull(getFunction())) output = TensorFlux.getOutput(function);
        return (M) output;
    }

    public Object getFunction() {
        if (Objects.nonNull(function)) return function;
        output = null;
        return function = compute();
    }

    public void forward() {
        for (Tensor o : getInput()) TensorFlux.computer(o);
        TensorFlux.forward(this);
    }

    public void backward() {
        TensorFlux.backward(this);
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        TensorFlux.reduce(this);
        for (Tensor o : getInput()) TensorFlux.reducer(o);
    }

}
