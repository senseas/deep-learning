package com.deep.framework.core;

import com.deep.framework.graph.*;

import java.io.Serializable;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.lang.Shape.fill;
import static com.deep.framework.lang.Shape.shape;

public class TensorFlux implements Serializable {
    public static List<Tensor> operators = new LinkedList<>();
    public static List<Tensor> params = new LinkedList<>();

    public static void intit(TensorExecutor executor) {
        forward(executor.getTensor());
        executor.setOperators(operators.toArray(Tensor[]::new));
        executor.setParams(params.toArray(Tensor[]::new));
        operators = params = null;
    }

    public static void forward(Tensor tensor) {
        if (operators.contains(tensor)) return;

        if (Objects.nonNull(tensor.getInput())) {
            for (Tensor o : tensor.getInput()) forward(o);
        }

        if (Objects.nonNull(tensor.getFunction())) {
            tensor.getFunction().forEach(TensorFlux::forward);
        }

        if ((Objects.nonNull(tensor.getInput()) || Objects.nonNull(tensor.getFunction()))) {
            operators.add(tensor);
        } else if (tensor.isReduce() && !params.contains(tensor)) {
            params.add(tensor);
        }
    }

    public static void concat(Tensor tensor) {
        if (Stream.of("Add", "Addx").anyMatch(a -> tensor.getName().contains(a))) {
            Stream<Tensor> stream = Stream.of();
            for (Tensor o : tensor.getInput()) {
                if (o.getName().equals(tensor.getName())) {
                    stream = Stream.concat(stream, Arrays.stream(o.getInput()));
                } else if (!(o instanceof TensorConst && o.data() == 0.0)) {
                    stream = Stream.concat(stream, Stream.of(o));
                }
            }
            tensor.setInput(stream.toArray(Tensor[]::new));
        }
    }

    public static <E> E getOutput(Object a) {
        Object c = fill(a, shape(Object.class, a), b -> {
            Tensor o = (Tensor) b;
            if (Objects.isNull(o.getShape())) return o;
            return o.getOutput();
        });
        return (E) fill(c, shape(Tensor.class, c), b -> b);
    }

    public static void syncOutputData(Tensor tensor) {
        final int[] i = {0};
        double[] data = tensor.getData();
        tensor.getFunction().forEach(a -> {
            if (a.getShape() != null) {
                a.getOutput().forEach(b -> data[i[0]++] = b.data());
            } else {
                data[i[0]++] = a.data();
            }
        });
    }

    public static void syncFunctionGrad(Tensor tensor) {
        final int[] i = {0};
        double[] grad = tensor.getGrad();
        tensor.getFunction().forEach(a -> {
            if (a.getShape() != null) {
                a.getOutput().forEach(b -> b.grad(grad[i[0]++]));
            } else {
                a.grad(grad[i[0]++]);
            }
        });
    }

}