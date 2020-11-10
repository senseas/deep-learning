package com.deep.framework.graph;

public class TensorOparetor extends Tensor {
    public TensorOparetor(String name, Tensor... input) {
        super(name, input);
    }

    public None getInput(int i) {
        Tensor input = getInput()[i];
        return (None) input.getOutput();
    }

    public None getOutput() {
        return (None) output;
    }

    public void execute() {
        if (getComputed()) return;
        Shape.farEach(getInput(), o -> ((Tensor) o).execute());
        Builder.computer(this);
        setComputed(true);
    }
}
