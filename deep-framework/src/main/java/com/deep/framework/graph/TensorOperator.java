package com.deep.framework.graph;

import com.deep.framework.framework.TensorFlux;
import com.deep.framework.lang.util.BeanUtil;

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
            this.valuex = zeros(shape);
            this.gradx = zeros(shape);
            this.reducex = booleans(shape);
            this.output = fillNones(this);
        } else {
            Arrays.fill((double[]) valuex, 0d);
            Arrays.fill((double[]) gradx, 0d);
            Arrays.fill((boolean[]) reducex, false);
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

    public String toString() {
        return new StringBuilder("extern \"C\"")
            .append("__global__ void ")
            .append(getName())
            .append("(double* inx , double* out)")
            .append("{")
            .append("  out[0] = ").append(BeanUtil.tmpl(getGrads(), getParams()))
            .append(";")
            .append("}").toString();
    }

}
