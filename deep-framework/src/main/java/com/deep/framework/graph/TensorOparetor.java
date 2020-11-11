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

    public void computeing() {
        if (getComputed()) return;
        Shape.farEach(getInput(), o -> ((Tensor) o).computeing());
        Builder.compute(this);
        setComputed(true);
    }

    public void gradienting() {
        if (!getComputed()) return;
        Builder.gradientCompute(this);
        setComputed(false);
        Shape.farEach(getInput(), o -> ((Tensor) o).gradienting());
    }

    public void reducer() {
        if (getComputed()) return;
        Builder.reducerFunction(this);
        Shape.farEach(getInput(), o -> ((Tensor) o).reducer());
        setComputed(true);
    }

}
