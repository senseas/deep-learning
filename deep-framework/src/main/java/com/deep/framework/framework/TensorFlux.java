package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

public class TensorFlux implements Serializable {
    static final double EX = 0.0000000001;

    public static void forward(Tensor tensor) {
        TensorExecutor.deep.getAndIncrement();
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.forward();
        });
        forwards(tensor);
        //CudaExecutor.compute(tensor);
        TensorExecutor.deep.getAndDecrement();
    }

    public static void backward(Tensor tensor) {
        TensorExecutor.deep.getAndIncrement();
        backwards(tensor);
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.backward();
        });
        //CudaExecutor.gradient(tensor);
        TensorExecutor.deep.getAndDecrement();
    }

    public static void reduce(Tensor tensor) {
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.reduce();
        });
    }

    public static void compute(Tensor tensor) {
        Object output = tensor.getOutput();
        if (Objects.nonNull(output)) {
            clearOutput(tensor);
            Object nones = tensor.compute();
            forEach(tensor.getOutput(), nones, (None out, None none) -> {
                out.setValue(none.getValue());
            });
        } else {
            Object nones = tensor.compute();
            createOutput(tensor, nones);
            forEach(tensor.getOutput(), nones, (None out, None none) -> {
                out.setValue(none.getValue());
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
        createOutput(tensor, nones);
        forEach(tensor.getOutput(), nones, (None out, None none) -> {
            out.setId(none.getId());
            out.setGradre(none.isGradre());
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

    public static void clearOutput(Tensor tensor) {
        if (BeanUtil.isTenser(tensor.getOutput())) {
            Arrays.fill((double[]) tensor.getValue(), 0d);
            Arrays.fill((double[]) tensor.getGrad(), 0d);
            Arrays.fill((boolean[]) tensor.getReduce(), false);
        } else {
            tensor.setValue(0d);
            tensor.setGrad(0d);
            tensor.setGradre(true);
            tensor.setReduce(false);
        }
    }

    public static void createOutput(Tensor tensor, Object o) {
        if (Objects.isNull((tensor.getOutput()))) {
            if (BeanUtil.isTenser(o) || BeanUtil.isArray(o)) {
                int[] shape = shapes(o);
                tensor.setShape(shape);
                tensor.setValue(zeros(shape));
                tensor.setGrad(zeros(shape));
                tensor.setReduce(booleans(shape));
                tensor.setGradre(true);
                tensor.setOutput(fillNones(tensor));
            } else {
                tensor.setValue(0d);
                tensor.setGrad(0d);
                tensor.setReduce(false);
                tensor.setGradre(true);
                tensor.setOutput(new None(tensor));
            }
        }
    }

    public static <E> E getOutput(Object a) {
        if (BeanUtil.isTenser(a)) {
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
        if (BeanUtil.isTenser(a)) {
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