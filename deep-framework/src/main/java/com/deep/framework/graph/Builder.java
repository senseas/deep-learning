package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

public class Builder extends Shape {

    public static void function(Tensor tensor) {
        if (BeanUtil.isNotOperation(tensor)) {
            Object function = tensor.compute();
            if (BeanUtil.isNotTensor(function)) {
                Tensor tense = (Tensor) function;
                if (BeanUtil.isNotOperation(tense)) {
                    tensor.setFunction(tense.getFunction());
                } else {
                    tensor.setFunction(function);
                }
            } else {
                tensor.setFunction(functions(function));
            }
        }
    }

    public static <M> M build(Tensor tensor, int i) {
        Tensor<Tensor> input = (Tensor) tensor.getInput()[i];
        if (BeanUtil.isOperation(tensor) && BeanUtil.isNone(input)) return (M) input.getOutput();
        if (BeanUtil.isOperation(tensor) && BeanUtil.isOperation(input)) return (M) input.getOutput();
        if (BeanUtil.isOperation(tensor)) return (M) input.getFunction().getOutput();
        if (BeanUtil.isOperation(input)) return (M) input;
        if (BeanUtil.isNoneNode(input)) return (M) tensors(input.getOutput());
        return (M) input.getFunction();
    }

}
