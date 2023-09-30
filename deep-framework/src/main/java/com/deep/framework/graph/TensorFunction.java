package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

public class TensorFunction extends Tensor {

    public TensorFunction(String name, Tensor... input) {
        super(name, input);
    }

    public TensorFunction(String name, int[] shape, Tensor... input) {
        super(name, input);
        this.shape = shape;
    }

    public TensorFunction(Tenser<Tensor> function) {
        super(null, new Tensor[0]);
        this.shape = function.shape;
        this.function = function;
        create(function);
    }

    public Tenser<Tensor> compute() { return null; }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        forEach(getFunction(), Tensor::forward);

        Object nones = TensorFlux.getOutput(getFunction());
        create(nones);
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> out.setValue(none.getValue()));
    }

    public void backward() {
        Object nones = TensorFlux.getOutput(getFunction());
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> none.setGrad(out.getGrad()));

        forEach(getFunction(), Tensor::backward);
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        forEach(getFunction(), Tensor::reduce);
        for (Tensor o : getInput()) o.reduce();
    }

    public void clearOutput() {
        if (Objects.nonNull(data)) {
            Arrays.fill(data, 0d);
            Arrays.fill(grads, 0d);
        }
    }

    public void clearGrad() {
        Arrays.fill(grads, 0d);
    }

    public void create(Object nones) {
        if (Objects.isNull(data)) {
            this.shape = shapes(nones);
            this.data = zeros(shape);
            this.grads = zeros(shape);
        }
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