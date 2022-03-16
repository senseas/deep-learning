package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

public class TensorFlux implements Serializable {
    static final double EX = 0.0000000001;

    public static void forward(Tensor tensor) {
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.forward();
        });
        forwards(tensor);
    }

    public static void backward(Tensor tensor) {
        backwards(tensor);
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.backward();
        });
    }

    public static void reduce(Tensor tensor) {
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.reduce();
        });
    }

    public static void compute(Tensor tensor) {
        Object nones = tensor.compute();
        if (!Objects.equals(nones, tensor.getOutput())) {
            tensor.zerosOutput(nones);
            forEach(tensor.getOutput(), nones, (None out, None none) -> {
                out.setValue(none.getValue());
                out.reset();
            });
        }
    }

    public static void computer(Tensor tensor) {
        if (Objects.nonNull(tensor.getOutput())) {
            forEach(tensor.getOutput(), (None out) -> {
                out.reset();
            });
        }
        tensor.forward();
    }

    public static void gradient(Tensor tensor) {
        tensor.gradient();
        forEach(tensor.getOutput(), (None out) -> {
            out.reset();
        });
    }

    public static void reducer(Tensor tensor) {
        if (tensor.isGradre()) {
            forEach(tensor.getOutput(), (None none) -> {
                if (!none.isReduce()) {
                    none.setReduce(true);
                    double valu = Math.abs(none.getValue()), grad = Math.abs(none.getGrad());
                    double rate = Math.min(valu / (grad + EX), grad / (valu + EX)) * TensorExecutor.rate;
                    double value = none.getValue() - rate * none.getGrad();
                    none.setValue(value);
                }
            });
        } else {
            tensor.reduce();
        }
    }

    private static void forwards(Tensor tensor) {
        Object nones = getOutput(tensor.getFunction());
        tensor.zerosOutput(nones);
        forEach(tensor.getOutput(), nones, (None out, None none) -> {
            out.setValue(none.getValue());
            out.reset();
        });
    }

    private static void backwards(Tensor tensor) {
        Object nones = getOutput(tensor.getFunction());
        forEach(tensor.getOutput(), nones, (None out, None none) -> {
            none.setGrad(out.getGrad());
        });
    }

    public static <E> E getOutput(Object a) {
        if (BeanUtil.isTensor(a)) {
            Object c = fill(a, shape(Object.class, a), b -> {
                Tensor o = (Tensor) b;
                return o.getOutput();
            });
            return (E) fill(c, shape(None.class, c), b -> b);
        } else {
            Tensor o = (Tensor) a;
            return o.getOutput();
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