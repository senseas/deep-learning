package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.IntStream;

import static com.deep.framework.lang.Shape.*;

public class TensorFunction extends Tensor {

    public TensorFunction(String name, Tensor... input) {
        super(name, input);
    }

    public TensorFunction(Tenser function) {
        super(null, new Tensor[0]);
        this.shape = function.shape;
        this.function = function;
        create(function);
    }

    public <M> M compute() { return null; }

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
        if (Objects.nonNull(data)) {
            Arrays.fill(grads, 0d);
        }
    }

    public void create(Object nones) {
        if (Objects.isNull(data)) {
            shape = shapes(nones);
            this.data = zeros(shape);
            this.grads = zeros(shape);
        }
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;

        function = new Tenser<>(Tensor.class, shape);
        return function = compute();
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        return TensorFlux.getTensor(input.getOutput());
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;

        output = new Tenser<>(Tensor.class, shape);
        IntStream.range(0, output.size()).forEach(i -> output.set(new Tensor(this, i), i));
        return output;
    }

}