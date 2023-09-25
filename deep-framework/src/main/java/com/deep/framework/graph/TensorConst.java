package com.deep.framework.graph;

public class TensorConst extends Tensor {

    public TensorConst(double value) { super(value, false); }

    public TensorConst(int[] shape, double value) { super(shape, value, false); }

}