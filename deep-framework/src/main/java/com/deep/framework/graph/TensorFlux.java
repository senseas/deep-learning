package com.deep.framework.graph;

import com.deep.framework.framework.Executor;
import com.deep.framework.lang.function.Func2;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

public class TensorFlux implements Serializable {
    static final double EX = 0.0000000001;

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

    public static void compute(Tensor tensor) {
        Object nones = tensor.compute(), outs = tensor.getOutput();
        if (Objects.nonNull(outs)) {
            farEach(nones, outs, (Func2<None>) (none, out) -> {
                out.reset();
                out.setValue(none.getValue());
            });
        } else {
            tensor.setOutput(nones);
        }
    }

    public static void computer(Tensor tensor) {
        if (BeanUtil.isNone(tensor)) {
            farEach(tensor.getOutput(), o -> {
                None out = (None) o;
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
            farEach(tensor.getOutput(), o -> {
                None none = (None) o;
                if (none.isGradre() && !none.isReduce()) {
                    none.setReduce(true);
                    double valu = Math.abs(none.getValue() + EX), grad = Math.abs(none.getGrad() + EX);
                    double rate = Math.min(valu / grad, grad / valu) * Executor.rate;
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
