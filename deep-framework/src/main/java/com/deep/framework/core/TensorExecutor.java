package com.deep.framework.core;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.util.Streams;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.core.TensorFlux.intit;
import static com.deep.framework.lang.ForEach.forBack;
import static com.deep.framework.lang.ForEach.forEach;

@Data
public class TensorExecutor implements Serializable {
    public static double rate = 0.003;
    private Tensor tensor;
    private Tensor input, inputx, label;
    private Tensor[] operators;
    private Tensor[] params;

    public TensorExecutor(Tensor tensor) {
        this.tensor = tensor;
    }

    public TensorExecutor(Tensor tensor, Tensor input, Tensor label) {
        this.tensor = tensor;
        this.input = input;
        this.label = label;
    }

    public TensorExecutor(Tensor tensor, Tensor input, Tensor inputx, Tensor label) {
        this.tensor = tensor;
        this.input = input;
        this.inputx = inputx;
        this.label = label;
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
        tensor.forward();
    }

    public void backward() {
        forEach(tensor, none -> none.grad(1d));
        tensor.backward();
    }

    public void reduce() {
        tensor.reducer();
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