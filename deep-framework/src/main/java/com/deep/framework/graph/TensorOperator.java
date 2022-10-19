package com.deep.framework.graph;

import com.deep.framework.framework.TensorFlux;

import java.util.Arrays;
import java.util.stream.Stream;

public class TensorOperator extends Tensor {

    public TensorOperator(String name, Tensor... input) {
        super(name, input);
        if (Arrays.asList("Add", "Addx").contains(name)) {
            Stream<Tensor> stream = Stream.of();
            for (Tensor o : input) {
                if (o.getName().equals(getName())) {
                    stream = Stream.concat(stream, Arrays.stream(o.getInput()));
                } else if (!((o instanceof TensorConst) && o.getValue().equals(0.0))) {
                    stream = Stream.concat(stream, Stream.of(o));
                }
            }
            setInput(stream.toArray(Tensor[]::new));
        }
    }

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

    public <M> M createOutput() {
        TensorFlux.createOutput(this, 1);
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
