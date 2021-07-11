package com.deep.framework.lang;

import com.jogamp.opencl.CLBuffer;

import java.nio.Buffer;

public class Tensor {
    protected Tensor[] input;
    protected float[] output;
    protected int[] shape;
    protected String name;
    protected CLBuffer<Buffer> buffer;

    public Tensor(int[] shape) {
        this.shape = shape;
        this.output = TensorUtil.random(shape);
    }

    public Tensor(int[] shape, float[] data) {
        this.shape = shape;
        this.output = data;
    }

    public Tensor(String name, Tensor... input) {
        this.name = name;
        this.input = input;
    }

    public <M> M compute(TensorContext context) { return null; }

    public void gradient(TensorContext context) { }

}
