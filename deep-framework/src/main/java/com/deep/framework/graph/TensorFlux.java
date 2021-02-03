package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class TensorFlux extends Shape {

    public static double rate = 0.003;

    public static void forward(Tensor tensor) {
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
            out.setGrad(null);
            out.setReduce(false);
            out.setValue(none.getValue());
        }
    }

    public static void computer(Tensor<None> tensor) {
        if (BeanUtil.isNone(tensor)) {
            tensor.getOutput().setGrad(null);
            tensor.getOutput().setReduce(false);
        } else {
            tensor.forward();
        }
    }

    public static void gradient(Tensor<None> tensor) {
        tensor.gradient();
    }

    public static void reducer(Tensor<None> tensor) {
        if (BeanUtil.isNone(tensor)) {
            None none = tensor.getOutput();
            if (BeanUtil.startsWithNone(tensor) && !none.isReduce()) {
                none.setReduce(true);
                double value = none.getValue() - rate * none.getGrad();
                none.setValue(value);
            }
        } else {
            tensor.reduce();
        }
    }

    public static <E> E getOutput(Object a) {
        if (BeanUtil.isTensor(a)) {
            return (E) fill(a, shape(Object.class, a), b -> {
                Tensor o = (Tensor) b;
                return o.getOutput();
            });
        } else {
            Tensor o = (Tensor) a;
            return (E) o.getOutput();
        }
    }

    public static <E> E getTensor(Object a) {
        if (BeanUtil.isTensor(a)) {
            return (E) fill(a, shape(Tensor.class, a), b -> {
                None o = (None) b;
                return new Tensor(o);
            });
        } else {
            None o = (None) a;
            return (E) new Tensor(o);
        }
    }

}
