package com.deep.framework.framework;

import com.deep.framework.graph.Tenser;
import lombok.Data;

@Data
public class Executor<E> extends Engine {
    private Tenser tenser;
    private Tenser input, label;

    public Executor(Tenser tenser) {
        this.tenser = tenser;
        graph(tenser);
    }

    public Executor(Tenser tenser, Tenser input, Tenser label) {
        this.tenser = tenser;
        this.input = input;
        this.label = label;
        graph(tenser);
    }

    public void run() {
        forward(tenser);
        backward(tenser);
    }

    public void run(E inSet, E labSet) {
        init(input, inSet);
        init(label, labSet);
        run();
    }
}
