package com.deep.framework.core;

import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorConst;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.lang.Shape.*;

public class TensorFlux implements Serializable {

    public static void concat(Tensor tensor) {
        if (Arrays.asList("Add", "Addx").contains(tensor.getName())) {
            Stream<Tensor> stream = Stream.of();
            for (Tensor o : tensor.getInput()) {
                if (o.getName().equals(tensor.getName())) {
                    stream = Stream.concat(stream, Arrays.stream(o.getInput()));
                } else if (!(o instanceof TensorConst && o.getValue() == 0.0)) {
                    stream = Stream.concat(stream, Stream.of(o));
                }
            }
            tensor.setInput(stream.toArray(Tensor[]::new));
        }
    }

    public static void createOutput(Tensor tensor, Object o) {
        if (Objects.isNull((tensor.getOutput()))) {
            if (BeanUtil.isTenser(o) || BeanUtil.isArray(o)) {
                int[] shape = shapes(o);
                tensor.setShape(shape);
                tensor.setData(zeros(shape));
                tensor.setGrad(zeros(shape));
                tensor.setOutput(fillNones(tensor));
            } else {
                tensor.setData(new double[]{0d});
                tensor.setGrads(new double[]{0d});
                tensor.setOutput(new Tenser<>(new Tensor[]{new Tensor(tensor, 0)}));
            }
        }
    }

    public static <E> E getOutput(Object a) {
        if (BeanUtil.isTenser(a)) {
            Object c = fill(a, shape(Object.class, a), b -> {
                Tensor o = (Tensor) b;
                return o.getOutput();
            });
            return (E) fill(c, shape(Tensor.class, c), b -> b);
        } else {
            Tensor o = (Tensor) a;
            return (E)o.getOutput();
        }
    }

    public static <E> E getTensor(Object a) {
        if (BeanUtil.isTenser(a)) {
            return (E) fill(a, shape(Tensor.class, a), b -> {
                Tensor o = (Tensor) b;
                return o;
            });
        } else {
            Tensor o = (Tensor) a;
            return (E) o;
        }
    }

}