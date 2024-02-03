package com.deep.framework.core;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tensers;
import com.deep.framework.lang.function.Func2;
import lombok.Data;

import java.io.Serializable;

import static com.deep.framework.lang.ForEach.forEach;

@Data
public class TensorExecutor<E> implements Serializable {
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
        tensor.forward();
        this.backward();
        tensor.reducer();
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
        Func2<Tensor, Double> func = Tensor::data;
        forEach(input.getOutput(), Tensers.tenser(o, input.getShape()), func);
    }

    public void setInputx(Object o) {
        Func2<Tensor, Double> func = Tensor::data;
        forEach(inputx.getOutput(), Tensers.tenser(o, inputx.getShape()), func);
    }

    public void setLabel(Object o) {
        Func2<Tensor, Double> func = Tensor::data;
        forEach(label.getOutput(), Tensers.tenser(o, label.getShape()), func);
    }

}
