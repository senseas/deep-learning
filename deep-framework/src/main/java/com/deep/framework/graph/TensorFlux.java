package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class TensorFlux extends Shape {

    public static double rate = 0.03;

    public static void forward(Tensor tensor) {
        BeanUtil.nameNode(tensor);
        farEach(tensor.getFunction(), o -> {
            Tensor a = (Tensor) o;
            a.forward();
        });
    }

    public static void backward(Tensor tensor) {
        farEach(tensor.getFunction(), o -> {
            Tensor a = (Tensor) o;
            a.backward();
        });
    }

    public static void reduce(Tensor tensor) {
        farEach(tensor.getFunction(), o -> {
            Tensor a = (Tensor) o;
            a.reduce();
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

    public static void computer(Tensor<None> tensor) {
        if (BeanUtil.isNone(tensor)) {
            tensor.getOutput().setReduce(false);
        } else {
            tensor.forward();
        }
    }

    public static void gradient(Tensor<None> tensor) {
        tensor.gradient();
        tensor.getOutput().setGrad(null);
    }

    public static void reducer(Tensor<None> tensor) {
        if (BeanUtil.isNone(tensor)) {
            None none = tensor.getOutput();
            if (BeanUtil.startsWithNone(tensor) && !none.getReduce()) {
                none.setReduce(true);
                Double value = none.getValue() - rate * none.getGrad();
                none.setValue(value);
            }
            none.setGrad(null);
        } else {
            tensor.reduce();
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
