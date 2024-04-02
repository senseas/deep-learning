package com.deep.framework.core;

import com.deep.framework.graph.Tensor;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.deep.framework.core.TensorFlux.intit;
import static com.deep.framework.lang.ForEach.forBack;
import static com.deep.framework.lang.ForEach.forEach;
import static com.deep.framework.lang.Shape.size;

@Data
public class TensorExecutor<E> implements Serializable {
    public static double rate = 0.003;
    public static final double eps = 0.0000001d;
    private Tensor tensor;
    private Tensor input, inputx, label;
    private Tensor[] operators;
    private Tensor[] params;

    public TensorExecutor(Tensor tensor) {
        this.tensor = tensor;
        intit(this);
    }

    public TensorExecutor(Tensor tensor, Tensor input, Tensor label) {
        this.tensor = tensor;
        this.input = input;
        this.label = label;
       intit(this);
    }

    public TensorExecutor(Tensor tensor, Tensor input, Tensor inputx, Tensor label) {
        this.tensor = tensor;
        this.input = input;
        this.inputx = inputx;
        this.label = label;
        intit(this);
    }

    public void run(E input, E label) {
        setInput(input);
        setLabel(label);
        run();
    }

    public void run(E input, E inputx, E label) {
        setInput(input);
        setInputx(inputx);
        setLabel(label);
        run();
    }

    public void run() {
        forward();
        backward();
        reduce();
    }

    public void forward(E input, E label) {
        setInput(input);
        setLabel(label);
        tensor.forward();
    }

    public void forward() {
        forEach(operators.length, i -> operators[i].forward());
        Stream.of(params).parallel().forEach(Tensor::forward);
    }

    public void backward() {
        tensor.getOutput().forEach(none -> none.grad(1d));
        forBack(operators.length, i -> operators[i].backward());
    }

    public void reduce() {
        Stream.of(params).parallel().forEach(Tensor::reducer);
    }

    public void setInput(Object o) {
        IntStream.range(0, size(input.getShape())).forEach(i -> input.getData()[i] = ((double[]) o)[i]);
    }

    public void setInputx(Object o) {
        IntStream.range(0, size(inputx.getShape())).forEach(i -> inputx.getData()[i] = ((double[]) o)[i]);
    }

    public void setLabel(Object o) {
        IntStream.range(0, size(label.getShape())).forEach(i -> label.getData()[i] = ((double[]) o)[i]);
    }

    public void clearFunction() {
        Stream.of(operators).forEach(o -> {
            if (Stream.of(o.getInput()).anyMatch(a -> a.isReduce() && Objects.isNull(a.getTensor()) || a.isStatus())) {
                o.setStatus(true);
            }
            if (Objects.nonNull(o.getFunction()) && !o.getFunction().data(0).isStatus()) {
                o.setOutput(null).setFunction(null);
            }
        });
    }

}