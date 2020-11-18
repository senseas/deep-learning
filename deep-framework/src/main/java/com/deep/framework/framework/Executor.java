package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.ForEach;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.function.Func2;
import lombok.Data;

import java.io.Serializable;

@Data
public class Executor<E> implements Serializable {

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


    public void init(Tensor a, Object b) {
        Func2<None, Double> func = (m, n) -> m.setValue(n);
        ForEach.farEach(a.getOutput(), b, func);
    }

    public void run() {
        tensor.forward();
        tensor.backward();
        tensor.reduce();
    }

    public void run(Func1 a) {
        tensor.forward();
        tensor.backward();
        a.apply(this);
        tensor.reduce();
    }

    public void run(Func1 a, Func1 b) {
        tensor.forward();
        tensor.backward();
        a.apply(this);
        tensor.reduce();
        b.apply(this);
    }

    public void run(E inSet, E labSet) {
        init(input, inSet);
        init(label, labSet);
        run();
    }

    public void run(E inSet, E labSet, Func1 a) {
        init(input, inSet);
        init(label, labSet);
        run(a);
    }

    public void run(E inSet, E labSet, Func1 a, Func1 b) {
        init(input, inSet);
        init(label, labSet);
        run(a, b);
    }
}
