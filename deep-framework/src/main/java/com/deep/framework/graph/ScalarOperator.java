package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.stream.Stream;

import static com.deep.framework.core.TensorFlux.concat;

public class ScalarOperator extends Tensor {

    public ScalarOperator(String name, Tensor... input) {
        super(name, input);
        concat(this);
        this.data = new double[1];
        this.grads = new double[1];
        this.output = new Tenser<>(new Tensor[]{new Tensor(this, 0)});
    }

    public double compute() { return 0; }

    public void gradient(double grad) { }

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        clearOutput();
        data[0] = compute();
    }

    public void backward() {
        gradient(grads[0]);
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        for (Tensor o : getInput()) o.reduce();
    }

    public void clearOutput() {
        data[0] = 0;
        grads[0] = 0;
    }

    public void clearGrad() {
        grads[0] = 0;
    }

    public Tensor getInput(int i) {
        return getInput()[i];
    }

    public Stream<Tensor> inputStream() {
        return Arrays.stream(getInput()).map(a -> a.getOutput().first());
    }

    public Tenser<Tensor> getOutput() {
        return output;
    }

}