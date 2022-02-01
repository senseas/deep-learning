package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Block;
import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.*;

import java.io.Serializable;
import java.nio.Buffer;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.Objects;

import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

public class TensorContext implements Serializable {

    private final Tensor tensor;

    private final CLContext context;

    private final CLCommandQueue queue;

    private final CLKernel compute, gradient;

    private CLBuffer<DoubleBuffer> valueBuffer, gradBuffer;

    private Block block;

    public TensorContext(Tensor tensor, CLContext context, CLCommandQueue queue, CLKernel compute, CLKernel gradient) {

        this.tensor = tensor;

        this.context = context;

        this.queue = queue;

        this.compute = compute;

        this.gradient = gradient;

    }

    private void setComputeArgs(final Object... params) {

        Arrays.stream(tensor.getInput()).forEach(o -> {

            CLBuffer buffer = o.getContext().getValueBuffer();

            queue.putWriteBuffer(buffer, true);

            compute.putArg(buffer);

        });

        CLBuffer buffer = getValueBuffer();

        queue.putWriteBuffer(buffer, true);

        compute.putArg(buffer);

        Arrays.stream(params).forEach(o -> setObjectConvert(compute, o));

        compute.rewind();
    }

    public void compute(final Object... params) {

        setComputeArgs(params);

        queue.put2DRangeKernel(compute, 0, 0, block.x, block.y, 0, 0);

        queue.putReadBuffer(getValueBuffer(), true);

        DoubleBuffer buffer = getValueBuffer().getBuffer();

        buffer.get((double[]) tensor.getValue());

        Arrays.stream(tensor.getInput()).forEach(tensor -> tensor.getContext().release());

        release();

    }

    private void setGradientArgs(final Object... params) {

        Arrays.stream(tensor.getInput()).forEach(o -> {

            CLBuffer buffer = o.getContext().getValueBuffer();

            queue.putWriteBuffer(buffer, true);

            gradient.putArg(buffer);

        });

        Arrays.stream(tensor.getInput()).forEach(o -> {

            CLBuffer buffer = o.getContext().getGradBuffer();

            queue.putWriteBuffer(buffer, true);

            gradient.putArg(buffer);

        });

        CLBuffer buffer = getGradBuffer();

        queue.putWriteBuffer(buffer, true);

        gradient.putArg(buffer);

        Arrays.stream(params).forEach(param -> setObjectConvert(gradient, param));

        gradient.rewind();

    }

    public void gradient(final Object... params) {

        setGradientArgs(params);

        queue.put2DRangeKernel(gradient, 0, 0, block.x, block.y, 0, 0);

        Arrays.stream(tensor.getInput()).forEach(o -> {

            CLBuffer<DoubleBuffer> clBuffer = o.getContext().getGradBuffer();

            queue.putReadBuffer(clBuffer, true);

            DoubleBuffer buffer = clBuffer.getBuffer();

            buffer.get((double[]) o.getGrad());

        });

        Arrays.stream(tensor.getInput()).forEach(tensor -> tensor.getContext().release());

        release();

    }

    public CLBuffer getBuffer(double[] arr) {

        Buffer directBuffer = Buffers.newDirectDoubleBuffer(arr);

        CLBuffer<Buffer> buffer = context.createBuffer(directBuffer, READ_WRITE);

        buffer.getBuffer().position(0);

        return buffer;

    }

    public CLBuffer<DoubleBuffer> getValueBuffer() {

        if (Objects.nonNull(valueBuffer)) return valueBuffer;

        return valueBuffer = getBuffer((double[]) tensor.getValue());

    }

    public CLBuffer<DoubleBuffer> getGradBuffer() {

        if (Objects.nonNull(gradBuffer)) return gradBuffer;

        return gradBuffer = getBuffer((double[]) tensor.getGrad());

    }

    public void release() {

        if (Objects.nonNull(valueBuffer)) {

            valueBuffer.release();

            valueBuffer = null;

        }

        if (Objects.nonNull(gradBuffer)) {

            gradBuffer.release();

            gradBuffer = null;

        }

    }

    public TensorContext setBlock(int... x) {
        if (x.length == 1) {
            this.block = new Block(x[0]);
        } else if (x.length == 2) {
            this.block = new Block(x[0], x[1]);
        } else if (x.length == 3) {
            this.block = new Block(x[0], x[1], x[2]);
        }
        return this;
    }

    public void setObjectConvert(CLKernel kernel, final Object value) {
        if (value instanceof CLMemory) {
            kernel.putArg((CLMemory) value);
        } else if (value instanceof Short) {
            kernel.putArg((Short) value);
        } else if (value instanceof Integer) {
            kernel.putArg((Integer) value);
        } else if (value instanceof Long) {
            kernel.putArg((Long) value);
        } else if (value instanceof Float) {
            kernel.putArg((Float) value);
        } else if (value instanceof Double) {
            kernel.putArg((Double) value);
        }
    }
}