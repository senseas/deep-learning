package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

import static com.deep.framework.core.TensorFlux.concat;

public class ScalarOperator extends Tensor {

    public ScalarOperator(String name, Tensor... input) {
        super(name, input);
        concat(this);
        this.data = new double[1];
        this.grad = new double[1];
    }

    public double compute() { return 0; }

    public void gradient(double grad) { }

    public void forward() {
        if(status) return;
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        data[0] = compute();
        status = true;
    }

    public void backward() {
        if(statusx) return;
        gradient(grad[0]);
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reducer() {
        if(statusx) return;
        for (Tensor o : getInput()) o.reducer();
        statusx = true;
    }

    public void clearOutput() {
        statusx = false;
        data[0] = 0;
        grad[0] = 0;
    }

    public void clearGrad() {
        status = false;
        grad[0] = 0;
    }

    public Tensor getInput(int i) {
        return getInput()[i];
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = new Tenser<>(new Tensor(this, 0));
    }

}