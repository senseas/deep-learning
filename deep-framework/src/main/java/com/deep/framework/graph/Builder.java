package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class Builder extends Shape {

    public static void function(Tensor tensor) {
        farEach(tensor.getFunction(), o -> {
            Tensor a = (Tensor) o;
            a.computeing();
        });
    }

    public static void gradientFunction(Tensor tensor) {
        farEach(tensor.getFunction(), o -> {
            Tensor a = (Tensor) o;
            a.gradienting();
        });
    }

    public static void compute(Tensor<None> tensor) {
        None none = tensor.compute(), out = tensor.getOutput();
        if (Objects.isNull(out)) {
            tensor.setOutput(none);
        } else {
            out.setValue(none.getValue());
        }
    }

    public static void gradientCompute(Tensor<None> tensor) {
        tensor.gradient();
        tensor.getOutput().setGrad(null);
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
