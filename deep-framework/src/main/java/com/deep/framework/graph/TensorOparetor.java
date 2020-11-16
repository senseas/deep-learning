package com.deep.framework.graph;

import com.deep.framework.framework.Sense;

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
            for (Tensor o : getInput()) Sense.computer(o);
            Sense.compute(this);
            setComputed("computed");
        }
    }

    public void backward() {
        if (!getComputed().equals("gradient")) {
            Sense.gradient(this);
            setComputed("gradient");
            for (Tensor o : getInput()) o.backward();
        }
    }

    public void reduce() {
        if (!getComputed().equals("reduce")) {
            for (Tensor o : getInput()) Sense.reducer(o);
            setComputed("reduce");
        }
    }

}
