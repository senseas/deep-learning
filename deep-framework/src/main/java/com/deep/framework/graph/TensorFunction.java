package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Stream;

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
        Tenser<Tensor> nones = TensorFlux.getOutput(getFunction());
        create();
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> out.data(none.data()));
        status = true;
    }

    public void backward() {
        Tenser<Tensor> nones = TensorFlux.getOutput(getFunction());
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> none.grad(out.grad()));

        forEach(getFunction(), Tensor::backward);
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reducer() {
        forEach(getFunction(), Tensor::reducer);
        for (Tensor o : getInput()) o.reducer();
    }

    public void clearOutput() {
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

    public void addInput(Tensor in) {
        Tensor[] tensors = Stream.concat(Stream.of(getInput()), Stream.of(in)).toArray(Tensor[]::new);
        setInput(tensors);
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        return output = Tensors(this);
    }

}