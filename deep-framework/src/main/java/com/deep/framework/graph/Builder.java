package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class Builder extends Shape {

    public static void function(Tensor tensor) {
        if (BeanUtil.isNotOperation(tensor)) {
            Object function = tensor.compute();
            if (Objects.nonNull(function)) {
                tensor.setFunction(functions(function));
            }
        }
    }

    public static <E> E getOutput(Object a) {
        if (BeanUtil.isTensor(a)) {
            return (E) fill(a, shape(None.class, a), b -> {
                Tensor o = (Tensor) b;
                return o.getOutput();
            });
        } else {
            Tensor o = (Tensor) a;
            return (E) o.getOutput();
        }
    }

}
