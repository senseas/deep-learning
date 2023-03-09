package com.deep.framework.graph;

import lombok.Data;

import static com.deep.framework.framework.TensorFlux.getOutput;
import static com.deep.framework.lang.ForEach.forEach;

@Data
public abstract class TensorFunctor {

    public TensorFunctor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        return input.getOutput();
    }

    public abstract <M> M compute();

    public abstract <M> M gradient(String grad);

    public String getGradId() {
        return out.getGradId();
    }

    public String getValId() {
        return out.getValId();
    }

    private None out;
    private String name = "Tensor::";
    private Tensor[] input;
}