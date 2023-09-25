package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.lang.Tenser;

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

    public Tenser<None> compute() { return null; }

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
        if (Objects.nonNull(value)) {
            Arrays.fill(value, 0d);
            Arrays.fill(grad, 0d);
        }
    }

    public void clearGrad() {
        Arrays.fill(grad, 0d);
    }

    public void create() {
        if (Objects.nonNull(value)) {
            this.value = random(shape);
            this.grad = zeros(shape);
            this.output = fillNones(this);
        }
    }

    public <M> M getInput(int i) {
        return getInput()[i].getOutput();
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

}