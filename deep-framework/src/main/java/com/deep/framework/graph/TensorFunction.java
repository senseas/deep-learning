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
        return (M) Shape.tensors(input.getOutput());
    }

    public Object getOutput() {
        if (Objects.nonNull(output)) return output;
        if (Objects.nonNull(getFunction())) output = Builder.getOutput(function);
        return output;
    }

    public Object getFunction() {
        if (Objects.nonNull(function)) return function;
        Object functions = compute();
        if (Objects.nonNull(functions)) function = Shape.functions(functions);
        return function;
    }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        Builder.forward(this);
    }

    public void backward() {
        Builder.backward(this);
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        Builder.reduce(this);
        for (Tensor o : getInput()) o.reduce();
    }

}
