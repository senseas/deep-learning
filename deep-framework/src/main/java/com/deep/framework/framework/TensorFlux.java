package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.util.BeanUtil;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.deep.framework.lang.Shape.*;

public class TensorFlux implements Serializable {
    static final double EX = 0.0000000001;

    @SneakyThrows
    public static void forward(Tensor tensor) {
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.forward();
        });
        forwards(tensor);
    }

    @SneakyThrows
    public static void backward(Tensor tensor) {
        backwards(tensor);
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.backward();
        });
        None output;
        if (BeanUtil.isTensor(tensor.getFunction())) {
            output = ((Tenser<None>) tensor.getOutput()).findFirst();
        } else {
            output = tensor.getOutput();
        }

        String parama = Arrays.stream(tensor.getInput()).map(a -> {
            return "double a" + ((None) a.getOutput()).getId();
        }).collect(Collectors.joining(","));

        String parame = Arrays.stream(tensor.getInput()).map(a -> {
            return "double e" + ((None) a.getOutput()).getId();
        }).collect(Collectors.joining(","));

        String code = Arrays.stream(tensor.getInput()).map(a -> {
            return ((None) a.getOutput()).getGradc();
        }).collect(Collectors.joining());

        String[] filex = output.getParam().split(",");
        String files = Arrays.stream(filex).limit(filex.length - 1).collect(Collectors.joining(","));

        String a =
        "class Tensor {\n" +
        "  double " + files + ";\n" +
        "  void compute(double a" + output.getId() + "," + parama + ") {\n" +
        "    " + output.getFunc() +
        "  }\n" +
        "  void gradient(double e" + output.getId() + "," + parame + ") {\n" +
        "    " + code +
        "  }\n" +
        "};\n";
        System.out.println(a);
    }

    public static void reduce(Tensor tensor) {
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.reduce();
        });
    }

    public static void compute(Tensor tensor) {
        Object output = tensor.getOutput();
        if (Objects.nonNull(output)) {
            resetOutput(tensor);
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
        /*forEach(tensor.getOutput(), (None out) -> {
            out.reset();
        });*/
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
            out.setFunc(none.getFunc());
            out.setFuncx(none.getFuncx());
            out.setParam(none.getParam());
            out.setValue(none.getValue());
            out.reset();
        });
    }

    private static void backwards(Tensor tensor) {
        Object nones = getOutput(tensor.getFunction());
        forEach(tensor.getOutput(), nones, (None out, None none) -> {
            none.setGradc(out.getGradc());
            none.setGradx(out.getGradx());
            none.setGrad(out.getGrad());
        });
    }

    public static void resetOutput(Tensor tensor) {
        if (BeanUtil.isTensor(tensor.getOutput())) {
            Arrays.fill((double[]) tensor.getValue(), 0d);
            Arrays.fill((double[]) tensor.getGrad(), 0d);
            Arrays.fill((boolean[]) tensor.getReduce(), false);
        } else {
            tensor.setValue(0d);
            tensor.setGrad(0d);
            tensor.setReduce(false);
        }
    }

    public static void createOutput(Tensor tensor, Object o) {
        if (Objects.isNull((tensor.getOutput()))) {
            if (BeanUtil.isTensor(o) || BeanUtil.isArray(o)) {
                int[] shape = shapes(o);
                tensor.setShape(shape);
                tensor.setValue(zeros(shape));
                tensor.setGrad(zeros(shape));
                tensor.setReduce(booleans(shape));
                tensor.setOutput(fillNones(tensor));
            } else {
                tensor.setValue(0d);
                tensor.setGrad(0d);
                tensor.setReduce(false);
                tensor.setOutput(new None(tensor));
            }
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