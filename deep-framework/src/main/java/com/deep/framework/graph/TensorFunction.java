package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

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
        if (output != null) return output;
        if (getFunction() != null) output = Builder.getOutput(function);
        return output;
    }
}
