package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.deep.framework.core.TensorFlux.concat;
import static com.deep.framework.lang.Shape.zeros;

public class TensorOperator extends Tensor {

    public TensorOperator(String name, Tensor... input) {
        super(name, input);
        concat(this);
    }

    public Tenser<Tensor> compute() { return null; }

    public void gradient() { }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        //create();
        compute();
    }

    public void backward() {
        gradient();
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
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

    public void create() {
        if (Objects.nonNull(data)) {
            this.data = zeros(shape);
            this.grads = zeros(shape);
        }
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public Stream inputStream() {
        return Arrays.stream(getInput()).map(Tensor::getOutput);
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;

        output = new Tenser<>(Tensor.class, shape);
        IntStream.range(0, output.size()).forEach(i -> output.set(new Tensor(this, i), i));
        return output;
    }

    public Tenser<Tensor> createOutput(Object o) {
        TensorFlux.createOutput(this, o);
        return getOutput();
    }

}