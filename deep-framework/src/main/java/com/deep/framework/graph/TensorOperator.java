package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.core.TensorFlux.concat;
import static com.deep.framework.lang.Shape.Tensors;
import static com.deep.framework.lang.Shape.zeros;

public class TensorOperator extends Tensor {

    public TensorOperator(String name, int[] shape, Tensor... input) {
        super(name, input);
        this.shape = shape;
        concat(this);
    }

    public Tenser<Tensor> compute() { return null; }

    public void gradient() { }

    public void forward() {
        if (status) return;
        for (Tensor o : getInput()) o.setParent(this).forward();

        clearOutput();
        create();
        compute();
        status = true;
    }

    public void backward() {
        if (!status) return;
        gradient();
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
        if (Objects.isNull(data)) return;
        Arrays.fill(data, 0d);
        Arrays.fill(grad, 0d);
    }

    public void clearGrad() {
        status = false;
        Arrays.fill(grad, 0d);
    }

    public void create() {
        if (Objects.nonNull(data)) return;
        this.data = zeros(shape);
        this.grad = zeros(shape);
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = Tensors(this);
    }

}