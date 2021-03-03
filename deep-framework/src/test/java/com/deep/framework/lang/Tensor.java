package com.deep.framework.lang;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLContext;

import java.nio.Buffer;
import java.util.Objects;

import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

public class Tensor {
    protected Tensor[] input;
    protected float[] data;
    protected int[] shape;
    protected String name;
    protected CLBuffer<Buffer> buffer;

    public Tensor(int[] shape) {
        this.shape = shape;
        this.data = TensorUtil.random(shape);
    }

    public Tensor(int[] shape, float[] data) {
        this.shape = shape;
        this.data = data;
    }

    public Tensor(String name, Tensor... input) {
        this.name = name;
        this.input = input;
    }

    public void compute(TensorContext kernel) { }

    public CLBuffer getBuffer(CLContext context) {
        if (Objects.nonNull(buffer)) return buffer;
        Buffer directBuffer = Buffers.newDirectFloatBuffer(this.data);
        return buffer = context.createBuffer(directBuffer, READ_WRITE);
    }

}
