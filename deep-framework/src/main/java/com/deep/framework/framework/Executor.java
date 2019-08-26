package com.deep.framework.framework;

import com.deep.framework.graph.Tenser;
import com.deep.framework.lang.function.Func1;
import lombok.Data;

@Data
public class Executor<E> extends Engine {
    private Tenser tenser;
    private Tenser input, label;

    public Executor(Tenser tenser) {
        this.tenser = tenser;
    }

    public Executor(Tenser tenser, Tenser input, Tenser label) {
        this.tenser = tenser;
        this.input = input;
        this.label = label;
    }

    public void run() {
        forward(tenser);
        backward(tenser);
    }

    public void run(Func1 a) {
        forward(tenser);
        a.apply(this);
        backward(tenser);
    }

    public void run(Func1 a, Func1 b) {
        forward(tenser);
        a.apply(this);
        backward(tenser);
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
