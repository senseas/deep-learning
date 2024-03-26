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
        if (status) return;
        for (Tensor o : getInput()) o.setParent(this).forward();

        clearOutput();
        data[0] = compute();
        status = true;
    }

    public void backward() {
        if (!status) return;
        gradient(grad[0]);
        clearGrad();

        for (Tensor o : getInput()) if (o.isParent(this)) o.setStatus(true).backward();
    }

    public void reducer() {
        if (states) return;
        for (Tensor o : getInput()) o.reducer();
        states = true;
    }

    public void clearOutput() {
        states = false;
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