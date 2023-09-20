package com.deep.framework.core;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorOperator;
import com.deep.framework.lang.util.BeanUtil;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

public class TensorFlux implements Serializable {
    static final double EX = 0.0000000001;

    @SneakyThrows
    public static void forward(Tensor tensor) {
        forEach(tensor.getFunction(), Tensor::forward);
        forwards(tensor);
        //CudaExecutor.compute(tensor);
    }

    @SneakyThrows
    public static void backward(Tensor tensor) {
        backwards(tensor);
        forEach(tensor.getFunction(), Tensor::backward);
        //CudaExecutor.gradient(tensor);
        forEach(tensor.getOutput(), None::reset);
    }

    public static void reduce(Tensor tensor) {
        forEach(tensor.getFunction(), Tensor::reduce);
    }

    public static void compute(TensorOperator tensor) {
        tensor.clearOutput();
        tensor.setOutput(tensor.compute());
    }

    public static void computer(Tensor tensor) {
        if (Objects.nonNull(tensor.getOutput())) {
            forEach(tensor.getOutput(), None::reset);
        }
        tensor.forward();
    }

    public static void gradient(TensorOperator tensor) {
        tensor.gradient();
        Arrays.fill(tensor.getGrad(), 0d);
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
            out.setValue(none.getValue());
            out.setGrad(0);
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
            Arrays.fill(tensor.getValue(), 0d);
            Arrays.fill(tensor.getGrad(), 0d);
        }
    }

    public static void createOutput(Tensor tensor, Object o) {
        if (Objects.isNull((tensor.getOutput()))) {
            if (BeanUtil.isTenser(o) || BeanUtil.isArray(o)) {
                int[] shape = shapes(o);
                tensor.setShape(shape);
                tensor.setValue(zeros(shape));
                tensor.setGrad(zeros(shape));
                tensor.setOutput(fillNones(tensor));
            } else {
                tensor.setValue(new double[]{0d});
                tensor.setGrad(new double[]{0d});
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