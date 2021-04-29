package com.deep.framework.graph;

import com.deep.framework.framework.Executor;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.util.BeanUtil;

import java.util.Objects;

public class TensorFlux extends Shape {

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
            farEach(nones, outs, (n, o) -> {
                None none = (None) n, out = (None) o;
                out.setGrad(null);
                out.setReduce(false);
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
                out.setGrad(null);
                out.setReduce(false);
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
                if (none.isEngrad() && !none.isReduce()) {
                    none.setReduce(true);
                    double value = none.getValue() - Executor.rate * none.getGrad();
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
