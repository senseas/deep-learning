package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.core.TensorFlux.syncFunctionGrad;
import static com.deep.framework.core.TensorFlux.syncOutputData;
import static com.deep.framework.lang.Shape.Tensors;
import static com.deep.framework.lang.Shape.zeros;

public class TensorFunction extends Tensor {

    public TensorFunction(String name, int[] shape, Tensor... input) {
        super(name, input);
        this.shape = shape;
    }

    public TensorFunction(Tenser<Tensor> function) {
        super("");
        this.shape = function.shape;
        this.function = function;
        create();
    }

    public Tenser<Tensor> compute() { return null; }

    public void forward() {
        for (Tensor o : getInput()) o.setRefer(this).forward();

        create();
        clearOutput();
        getFunction().forEach(Tensor::forward);
        syncOutputData(this);
    }

    public void backward() {
        syncFunctionGrad(this);
        getFunction().forEach(Tensor::backward);
        clearGrad();

        for (Tensor o : getInput()) o.setRefer(this).backward();
    }

    public void reducer() {
        getFunction().forEach(Tensor::reducer);
        for (Tensor o : getInput()) o.setRefer(this).reducer();
    }

    public void clearOutput() {
        Arrays.fill(data, 0d);
        Arrays.fill(grad, 0d);
    }

    public void clearGrad() {
        Arrays.fill(grad, 0d);
    }

    public void create() {
        if (Objects.nonNull(data)) return;
        this.data = zeros(shape);
        this.grad = zeros(shape);
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = compute();
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = Tensors(this);
    }

}