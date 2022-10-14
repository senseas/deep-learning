package com.deep.framework.graph;

import com.deep.framework.framework.TensorFlux;
import lombok.Data;

@Data
public abstract class TensorFunctor {

    public TensorFunctor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        if (input instanceof TensorFunction) return TensorFlux.getOutput(input.getFunction());
        return input.getOutput();
    }

    public abstract <M> M compute();

    public abstract <M> M gradient(String grad);

    public String getGradId() {
        return "  e" + id + "  ";
    }

    public String getValId() {
        return "  a" + id + "  ";
    }

    private int id;
    private String name = "Tensor::";
    private Tensor[] input;
}