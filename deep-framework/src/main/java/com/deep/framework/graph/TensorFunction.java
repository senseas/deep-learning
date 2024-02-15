package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.core.TensorFlux.syncFunctionGrad;
import static com.deep.framework.core.TensorFlux.syncOutputData;
import static com.deep.framework.lang.Shape.*;

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
        if (status) return;
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        forEach(getFunction(), Tensor::forward);

        create();
        syncOutputData(this);
        status = true;
    }

    public void backward() {
        if (statusx) return;
        for (Tensor o : getInput()) o.setStatus(o.statusx).setStatusx(true);

        syncFunctionGrad(this);
        forEach(getFunction(), Tensor::backward);
        clearGrad();

        for (Tensor o : getInput()) o.setStatusx(o.status).setStatus(false).backward();
    }

    public void reducer() {
        if(statusx) return;
        forEach(getFunction(), Tensor::reducer);
        for (Tensor o : getInput()) o.reducer();
        statusx = true;
    }

    public void clearOutput() {
        statusx = false;
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