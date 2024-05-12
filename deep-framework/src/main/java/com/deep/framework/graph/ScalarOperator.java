package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

import static com.deep.framework.core.TensorFlux.concat;

public class ScalarOperator extends Tensor {

    public ScalarOperator(String name, Tensor... input) {
        super(name, new int[]{1}, input);
        concat(this);
    }

    public double compute() { return 0; }

    public void gradient(double grad) { }

    public void forward() {
        clearOutput();
        create();
        data[0] = compute();
    }

    public void backward() {
        gradient(grad[0]);
        clearGrad();
    }

    private void clearOutput() {
        if (Objects.isNull(data)) return;
        data[0] = 0;
        grad[0] = 0;
    }

    private void clearGrad() {
        grad[0] = 0;
    }

    private void create() {
        if (Objects.nonNull(data)) return;
        this.data = new double[1];
        this.grad = new double[1];
    }

    public Tensor getInput(int i) {
        return getInput()[i];
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = new Tenser<>(this);
    }

}