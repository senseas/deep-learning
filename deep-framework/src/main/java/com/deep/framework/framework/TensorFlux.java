package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.annotation.Cuda;
import com.deep.framework.lang.util.BeanUtil;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.lang.annotation.Annotation;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static com.deep.framework.lang.Shape.*;

public class TensorFlux implements Serializable {
    static final double EX = 0.0000000001;

    @SneakyThrows
    public static void forward(Tensor tensor) {
        TensorExecutor.deep.getAndIncrement();
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.forward();
        });
        Annotation cuda = tensor.getClass().getMethod("compute").getAnnotation(Cuda.class);
        if (Objects.nonNull(cuda)) {
            CudaExecutor.compute(tensor);
        }
        forwards(tensor);
        TensorExecutor.deep.getAndDecrement();
    }

    @SneakyThrows
    public static void backward(Tensor tensor) {
        TensorExecutor.deep.getAndIncrement();
        backwards(tensor);
        forEach(tensor.getFunction(), (Tensor a) -> {
            a.backward();
        });
        TensorExecutor.deep.getAndDecrement();
        Annotation cuda = tensor.getClass().getMethod("compute").getAnnotation(Cuda.class);
        if (Objects.nonNull(cuda)) {
            CudaExecutor.gradient(tensor);
        }
        if (TensorExecutor.status) {
            List<String> parama = new ArrayList(), parame = new ArrayList(), paramg = new ArrayList();
            Arrays.stream(tensor.getInput()).forEach(a -> {
                forEach(a.getOutput(), (None o) -> {
                    parama.add("double a" + o.getId());
                    parame.add("double e" + o.getId());
                    paramg.addAll(List.of(o.getGradc().split(";")));
                });
            });

            List<String> paramf = new ArrayList(), paramc = new ArrayList(), paramp = new ArrayList();
            forEach(tensor.getOutput(), (None a) -> {
                paramf.add(a.getParam());
                paramp.add("double e" + a.getId());
                paramc.addAll(List.of(a.getFunc().split(";")));
            });

            String paramcx = paramc.stream().distinct().collect(Collectors.joining(";")) + ";";
            String paramgx = paramg.stream().distinct().collect(Collectors.joining(";")) + ";";
            String code =
            "class Tensor {\n" +
            "private:\n"+
            "  double " + String.join("", paramf) + ";\n" +
            "public:\n"+
            "  void compute(" + String.join(",", parama) + ") {\n" +
            "    " + paramcx +
            "\n  }\n" +
            "  void gradient("+ String.join(",", parama) +","+ String.join(",", paramp) +"," + String.join(",", parame) + ") {\n" +
            "    " + paramgx +
            "\n  }\n" +
            "};";
            //System.out.println(code);
        }
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
        if (TensorExecutor.status&&TensorExecutor.deep.get()>1) {
            forEach(tensor.getOutput(), nones, (None out, None none) -> {
                out.setId(none.getId());
                out.setFunc(none.getFunc());
                out.setFuncx(none.getFuncx());
                out.setParam(none.getParam());
            });
        }
        forEach(tensor.getOutput(), nones, (None out, None none) -> {
            out.setValue(none.getValue());
            out.reset();
        });
    }

    private static void backwards(Tensor tensor) {
        Object nones = getOutput(tensor.getFunction());
        if (TensorExecutor.status&&TensorExecutor.deep.get()>1) {
            forEach(tensor.getOutput(), nones, (None out, None none) -> {
                none.setGradc(out.getGradc());
                none.setGradx(out.getGradx());
            });
        }
        forEach(tensor.getOutput(), nones, (None out, None none) -> {
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