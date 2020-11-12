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
        if (getComputed().equals("computed")) return;
        Shape.farEach(getInput(), o -> Builder.computer((Tensor) o));
        Builder.compute(this);
        setComputed("computed");
    }

    public void gradienting() {
        if (getComputed().equals("gradient")) return;
        Builder.gradientCompute(this);
        setComputed("gradient");
        Shape.farEach(getInput(), o -> ((Tensor) o).gradienting());
    }

    public void reduceing() {
        if (getComputed().equals("reduce")) return;
        Shape.farEach(getInput(), o -> Builder.reducer((Tensor) o));
        setComputed("reduce");
    }

}