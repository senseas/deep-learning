package com.deep.framework.graph;

import com.deep.framework.framework.TensorFlux;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.lang.Shape.*;

public class TensorOperator extends Tensor {

    public TensorOperator(String name, Tensor... input) {
        super(name, input);
        if (Arrays.asList("Add", "Addx").contains(name)) {
            Stream<Tensor> stream = Stream.of();
            for (Tensor o : input) {
                Stream<Tensor> children = o.getName().equals(getName()) ? Arrays.stream(o.getInput()) : Stream.of(o);
                stream = Stream.concat(stream, children);
            }
            setInput(stream.toArray(Tensor[]::new));
        }
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        return input.getOutput();
    }

    public Stream inputStream() {
        return Arrays.stream(getInput()).map(input -> input.getOutput());
    }

    public <M> M getOutput() {
        return (M) output;
    }

    public <M> M createOutput(Object o) {
        if (Objects.isNull(getOutput())) {
            this.shape = shapes(o);
            this.value = zeros(shape);
            this.grad = zeros(shape);
            this.reduce = booleans(shape);
            this.output = fillNones(this);
        } else {
            Arrays.fill((double[]) value, 0d);
            Arrays.fill((double[]) grad, 0d);
            Arrays.fill((boolean[]) reduce, false);
        }
        return getOutput();
    }

    public <M> M createOutput() {
        if (Objects.isNull(getOutput())) {
            this.value = 0d;
            this.grad = 0d;
            this.reduce = false;
            this.output = new None(this);
        } else {
            this.value = 0d;
            this.grad = 0d;
            this.reduce = false;
        }
        return getOutput();
    }

    public void forward() {
        for (Tensor o : getInput()) TensorFlux.computer(o);
        TensorFlux.compute(this);
    }

    public void backward() {
        TensorFlux.gradient(this);
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        for (Tensor o : getInput()) TensorFlux.reducer(o);
    }

}
