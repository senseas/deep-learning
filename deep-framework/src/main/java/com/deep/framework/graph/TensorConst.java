package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Objects;
import java.util.stream.IntStream;

public class TensorConst extends Tensor {

    public TensorConst(double value) { super(value); }

    public TensorConst(Tensor tensor, int idx) { super(tensor,idx); }

    public TensorConst(int[] shape, double value) { super(shape, value); }

    public void forward() { }

    public void backward() { }

    public void reduce() { }

    public void setGrad(double grad) {}

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;

        output = new Tenser<>(Tensor.class, shape);
        IntStream.range(0, output.size()).forEach(i -> output.set(new TensorConst(this, i), i));
        return output;
    }
}