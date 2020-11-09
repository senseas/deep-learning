package com.deep.framework.graph;

public class TensorOparetor extends Tensor {
    public TensorOparetor(String name, Tensor... input) {
        super(name, input);
    }

    public None getInput(int i) {
        Tensor input = getInput()[i];
        Builder.computer(input);
        return (None) input.getOutput();
    }

    public None getOutput() {
        return (None) output;
    }

    public void execute() {
        Shape.farEach(getInput(), o -> {
            Tensor tensor = (Tensor) o;
            tensor.execute();
        });
        Builder.computer(this);
    }
}
