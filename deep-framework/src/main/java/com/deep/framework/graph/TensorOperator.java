package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;

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
                if (o.getName().equals(getName())) {
                    stream = Stream.concat(stream, Arrays.stream(o.getInput()));
                } else if (!(o instanceof TensorConst && o.getValue()[0] == 0.0)) {
                    stream = Stream.concat(stream, Stream.of(o));
                }
            }
            setInput(stream.toArray(Tensor[]::new));
        }
    }

    public double[] compute() { return null; }

    public void gradient() { }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        return input.getOutput();
    }

    public Stream inputStream() {
        return Arrays.stream(getInput()).map(Tensor::getOutput);
    }

    public <M> M getOutput() {
        return (M) output;
    }

    public <M> M createOutput(Object o) {
        TensorFlux.createOutput(this, o);
        return getOutput();
    }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        create();
        value = compute();
    }

    public void backward() {
        gradient();
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        for (Tensor o : getInput()) o.reduce();
    }

    public void create() {
        if (Objects.isNull(value)) {
            this.value = random(shape);
            this.grad = zeros(shape);
            this.output = fillNones(this);
        }
    }

    public void clearOutput() {
        if (Objects.isNull(value)) return;
        Arrays.fill(value, 0d);
        Arrays.fill(grad, 0d);
    }

    public void clearGrad() {
        Arrays.fill(grad, 0d);
    }
}
