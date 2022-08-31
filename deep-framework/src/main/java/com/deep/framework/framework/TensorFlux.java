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
        farEach(tensor.getFunction(), (Tensor a) -> {
            a.forward();
        });
    }

    public static void backward(Tensor tensor) {
        farEach(tensor.getFunction(), (Tensor a) -> {
            a.backward();
        });
    }

    public static void reduce(Tensor tensor) {
        farEach(tensor.getFunction(), (Tensor a) -> {
            a.reduce();
        });
    }

    public static void compute(Tensor tensor) {
        Object outs = tensor.getOutput();
        if (Objects.nonNull(outs)) {
            farEach(outs, (None out) -> {
                out.setValue(0);
                out.reset();
            });
            Object nones = tensor.compute();
            if (!Objects.equals(nones, outs)) {
                farEach(nones, outs, (None none, None out) -> {
                    out.setValue(none.getValue());
                });
            }
        } else {
            Object nones = tensor.compute();
            tensor.setOutput(nones);
        }
    }

    public static void computer(Tensor tensor) {
        if (BeanUtil.isNone(tensor)) {
            farEach(tensor.getOutput(), (None out) -> {
                out.reset();
            });
        } else {
            tensor.forward();
        }
    }

    public static void gradient(Tensor tensor) {
        tensor.gradient();
    }

    public static void reducer(Tensor tensor) {
        if (BeanUtil.isNone(tensor)) {
            farEach(tensor.getOutput(), (None none) -> {
                if (none.isGradre() && !none.isReduce()) {
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
