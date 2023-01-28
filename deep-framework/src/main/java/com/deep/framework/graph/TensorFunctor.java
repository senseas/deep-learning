package com.deep.framework.graph;

import com.deep.framework.framework.TensorCore;
import com.deep.framework.framework.TensorFlux;
import lombok.Data;

import static com.deep.framework.framework.TensorCore.*;

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
        if (getName().startsWith("None")) return "inGrad[idx * N +" + idxInGrad.getAndIncrement() + "]";
        if (root) return "outGrad[idx * M +" + idxOutGrad.getAndIncrement() + "]";
        return "e" + id;
    }

    public String getValId() {
        if (getName().startsWith("None")) return "in[idx * N +" + idxIn.getAndIncrement() + "]";
        return out.getValId();
    }

    private TensorCore core;
    private int id;
    private None out;
    private boolean root;
    private String name = "Tensor::";
    private Tensor[] input;
}