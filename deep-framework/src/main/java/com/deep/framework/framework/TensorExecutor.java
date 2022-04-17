package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.function.Func21;
import lombok.Data;

import java.io.Serializable;

import static com.deep.framework.lang.ForEach.farEach;
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
        tensor.reduce();
        System.out.println(System.currentTimeMillis() - ss);
    }

    public void forward(E input, E label) {
        setInput(input);
        setLabel(label);
        tensor.forward();
    }

    public void backward() {
        forEach(tensor.getOutput(), (None a) -> a.setGrad(1d));
        tensor.backward();
    }

    public void reduce() {
        tensor.reduce();
    }

    public void setInput(Object o) {
        Func21<None, Double> func = (m, n) -> m.setValue(n);
        farEach(input.getOutput(), o, func);
    }

    public void setLabel(Object o) {
        Func21<None, Double> func = (m, n) -> m.setValue(n);
        farEach(label.getOutput(), o, func);
    }

}
