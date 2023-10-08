package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.core.TensorFlux.concat;
import static com.deep.framework.lang.Shape.Tensors;
import static com.deep.framework.lang.Shape.zeros;

public class TensorOperator extends Tensor {

    public TensorOperator(String name, Tensor... input) {
        super(name, input);
        concat(this);
    }

    public TensorOperator(String name, int[] shape, Tensor... input) {
        super(name, input);
        this.shape = shape;
        concat(this);
    }

    public Tenser<Tensor> compute() { return null; }

    public void gradient() { }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        create();
        compute();
    }

    public void backward() {
        gradient();
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reducer() {
        for (Tensor o : getInput()) o.reducer();
    }

    public void clearOutput() {
        if (Objects.nonNull(data)) {
            Arrays.fill(data, 0d);
            Arrays.fill(grad, 0d);
        }
    }

    public void clearGrad() {
        Arrays.fill(grad, 0d);
    }

    public void create() {
        if (Objects.isNull(data) && Objects.nonNull(shape)) {
            this.data = zeros(shape);
            this.grad = zeros(shape);
        }
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output) || Objects.isNull(shape)) return output;
        return output = Tensors(this);
    }

    public Tenser<Tensor> createOutput(Object o) {
        TensorFlux.createOutput(this, o);
        return getOutput();
    }

}