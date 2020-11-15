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

    public void forward() {
        if (!getComputed().equals("computed")) {
            Shape.farEach(getInput(), o -> Builder.computer((Tensor) o));
            Builder.compute(this);
            setComputed("computed");
        }
    }

    public void backward() {
        if (!getComputed().equals("gradient")) {
            Builder.gradient(this);
            setComputed("gradient");
            Shape.farEach(getInput(), o -> ((Tensor) o).backward());
        }
    }

    public void reduce() {
        if (!getComputed().equals("reduce")) {
            Shape.farEach(getInput(), o -> Builder.reducer((Tensor) o));
            setComputed("reduce");
        }
    }

}
