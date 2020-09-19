package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class Builder extends Shape {

    public static void function(Tensor tensor) {
        if (BeanUtil.isNotOperation(tensor)) {
            Object function = tensor.compute();
            if (Objects.nonNull(function)) {
                tensor.setFunction(functions(function));
                tensor.setOutput(outputs(function));
            }
        }
    }

    public static <M> M getInput(Tensor tensor, int i) {
        Tensor<Tensor> input = (Tensor) tensor.getInput()[i];
        if (BeanUtil.isOperation(tensor)) {
            if (BeanUtil.isNone(input)) return (M) input.getOutput();
            if (BeanUtil.isOperation(input)) return (M) input.getOutput();
            return (M) input.getOutput();
        } else {
            if (BeanUtil.isOperation(input)) return (M) input;
            if (BeanUtil.isNoneNode(input)) return (M) tensors(input.getOutput());
            return (M) input.getFunction();
        }
    }

}
