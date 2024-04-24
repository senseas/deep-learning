package com.deep.framework.core;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import lombok.Data;

import java.io.Serializable;
import java.util.Arrays;

@Data
public class TensorExecutor implements Serializable {
    public static double rate = 0.003;
    public static final double eps = 0.0000001d;
    private Tensor tensor;
    private Tensor input, inputx, label;

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
        tensor.forward();
        this.backward();
        tensor.reducer();
    }

    public void forward(double[] input, double[] label) {
        setInput(input);
        setLabel(label);
        tensor.forward();
    }

    public void backward() {
        Arrays.fill(tensor.getGrad(), 1d);
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
