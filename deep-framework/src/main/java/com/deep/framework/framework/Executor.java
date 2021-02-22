package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.ForEach;
import com.deep.framework.lang.function.Func2;
import lombok.Data;

import java.io.Serializable;

@Data
public class Executor<E> implements Serializable {
    public static double rate = 0.003;
    private Tensor tensor;
    private Tensor input, label;

    public Executor(Tensor tensor) {
        this.tensor = tensor;
    }

    public Executor(Tensor tensor, Tensor input, Tensor label) {
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
        tensor.forward();
        tensor.backward();
        tensor.reduce();
    }

    public void forward(E input, E label) {
        setInput(input);
        setLabel(label);
        tensor.forward();
    }

    public void setInput(Object o) {
        Func2<None, Double> func = (m, n) -> m.setValue(n);
        ForEach.farEach(input.getOutput(), o, func);
    }

    public void setLabel(Object o) {
        Func2<None, Double> func = (m, n) -> m.setValue(n);
        ForEach.farEach(label.getOutput(), o, func);
    }

}
