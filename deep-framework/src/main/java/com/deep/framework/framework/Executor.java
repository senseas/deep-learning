package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.function.Func1;
import lombok.Data;
import org.apache.log4j.Logger;

@Data
public class Executor<E> extends Engine {
    Logger log = Logger.getLogger(Executor.class);

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

    public void run() {
        forward(tensor);
        backward(tensor);
    }

    public void run(Func1 a) {
        tensor.execute();
        // forward(tensor);
        a.apply(this);
        backward(tensor);
    }

    public void run(Func1 a, Func1 b) {
        forward(tensor);
        a.apply(this);
        backward(tensor);
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
