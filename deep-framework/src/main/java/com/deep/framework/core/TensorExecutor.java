package com.deep.framework.core;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import lombok.Data;

import java.io.Serializable;
import java.util.stream.Stream;

import static com.deep.framework.core.TensorFlux.intit;
import static com.deep.framework.lang.ForEach.forBack;
import static com.deep.framework.lang.ForEach.forEach;

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

    public void run(double[] input, double[] label) {
        setInput(input);
        setLabel(label);
        run();
    }

    public void run(double[] input, double[] inputx, double[] label) {
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

    public void forward(double[] input, double[] label) {
        setInput(input);
        setLabel(label);
        tensor.forward();
    }

    public void forward() {
        forEach(operators.length, i -> operators[i].forward());
        Stream.of(params).forEach(Tensor::forward);
    }

    public void backward() {
        forEach(tensor, none -> none.grad(1d));
        forBack(operators.length, i -> operators[i].backward());
    }

    public void reduce() {
        Stream.of(params).forEach(Tensor::reducer);
    }

    public void setInput(double[] data) {
        Shape.copy(data, input.getData());
    }

    public void setInputx(double[] data) {
        Shape.copy(data, inputx.getData());
    }

    public void setLabel(double[] data) {
        Shape.copy(data, label.getData());
    }

}