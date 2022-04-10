package com.deep.framework.graph;

import com.deep.framework.framework.TensorFlux;
import com.deep.framework.lang.util.BeanUtil;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Stream;

import static com.deep.framework.lang.ForEach.farEach;
import static com.deep.framework.lang.Shape.zeroNones;

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
        if (BeanUtil.isFunction(input)) return TensorFlux.getOutput(input.getFunction());
        return input.getOutput();
    }

    public Stream inputStream() {
        return Arrays.stream(getInput()).parallel().map(input -> BeanUtil.isFunction(input) ?
                TensorFlux.getOutput(input.getFunction()) : input.getOutput());
    }

    public <M> M getOutput() {
        return (M) output;
    }

    public <M> M createOutput(Object o) {
        if (Objects.isNull(getOutput())) {
            this.output = zeroNones(o);
        } else {
            farEach(getOutput(), (None out) -> {out.setValue(0d);out.reset();});
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
