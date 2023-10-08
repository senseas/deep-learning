package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

import static com.deep.framework.lang.Shape.TensorConsts;

public class TensorConst extends Tensor {

    public TensorConst(double value) { super(value); }

    public TensorConst(double value,int[] shape) { super(shape, value); }

    public TensorConst(Tensor tensor, int idx) { super(tensor, idx); }

    public void forward() {}

    public void reducer() {}

    public void grad(double grad) {}

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output) || Objects.isNull(shape)) return output;
        return output = TensorConsts(this);
    }
}