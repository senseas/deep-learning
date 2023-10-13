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
    private Tensor tensor;
    private Tensor input, label;

    public TensorExecutor(Tensor tensor) {
        this.tensor = tensor;
    }

    public TensorExecutor(Tensor tensor, Tensor input, Tensor label) {
        this.tensor = tensor;
        this.input = input;
        this.label = label;
    }

    public void run(E input, E label) {
        setInput(input);
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
        Func2<Tensor, Double> func = Tensor::data;
        forEach(input.getOutput(), Tensers.tenser(o), func);
    }

    public void setLabel(Object o) {
        Func2<Tensor, Double> func = Tensor::data;
        forEach(label.getOutput(), Tensers.tenser(o), func);
    }

}
