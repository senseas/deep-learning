package com.deep.framework.core;

import com.deep.framework.graph.Tensor;
import lombok.Data;

import java.io.Serializable;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.forEach;

@Data
public class TensorExecutor<E> implements Serializable {
    public static double rate = 0.003;
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
        long ss = System.currentTimeMillis();
        tensor.forward();
        this.backward();
        tensor.reducer();
        //System.out.println(System.currentTimeMillis() - ss);
    }

    public void forward(E input, E label) {
        setInput(input);
        setLabel(label);
        tensor.forward();
    }

    public void backward() {
        forEach(tensor.getOutput(), (Tensor none) -> none.grad(1d));
        tensor.backward();
    }

    public void reduce() {
        tensor.reducer();
    }

    public void setInput(Object o) {
        IntStream.range(0, input.getData().length).forEach(i -> input.getData()[i] = ((double[]) o)[i]);
    }

    public void setInputx(Object o) {
        IntStream.range(0, inputx.getData().length).forEach(i -> inputx.getData()[i] = ((double[]) o)[i]);
    }

    public void setLabel(Object o) {
        IntStream.range(0, label.getData().length).forEach(i -> label.getData()[i] = ((double[]) o)[i]);
    }

}
