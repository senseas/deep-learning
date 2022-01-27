package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Block;
import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.*;

import java.io.Serializable;
import java.nio.Buffer;
import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

public class TensorContext implements Serializable {

    private final Tensor tensor;
    private final CLContext context;
    private final CLCommandQueue queue;
    private final CLKernel compute, gradient;
    private List<CLBuffer<DoubleBuffer>> input;
    private CLBuffer<DoubleBuffer> output;
    private Block block;

    public TensorContext(Tensor tensor, CLContext context, CLCommandQueue queue, CLKernel compute, CLKernel gradient) {
        this.tensor = tensor;
        this.context = context;
        this.queue = queue;
        this.compute = compute;
        this.gradient = gradient;
    }

    private void setComputeArgs(final Object... params) {

        for (Tensor tensor : tensor.getInput()) {
            CLBuffer buffer = getBuffer((double[]) tensor.getValue());
            queue.putWriteBuffer(buffer, true);
            compute.putArg(buffer);
        }

        output = getBuffer((double[]) tensor.getValue());
        queue.putWriteBuffer(output, true);
        compute.putArg(output);

        for (Object param : params) {
            setObjectConvert(compute, param);
        }

        compute.rewind();
    }

    public void compute(final Object... params) {
        setComputeArgs(params);
        queue.put2DRangeKernel(compute, 0, 0, block.x, block.y, 0, 0);
        queue.putReadBuffer(output, true);
        DoubleBuffer buffer = output.getBuffer();
        double[] value = (double[]) tensor.getValue();
        IntStream.range(0, value.length).forEach(i -> value[i] = buffer.get(i));
        output.release();
    }

    private void setGradientArgs(final Object... params) {
        input = new ArrayList();

        for (Tensor tensor : tensor.getInput()) {
            CLBuffer valueBuffer = getBuffer((double[]) tensor.getValue());
            queue.putWriteBuffer(valueBuffer, true);
            gradient.putArg(valueBuffer);

            CLBuffer gradBuffer = getBuffer((double[]) tensor.getGrad());
            queue.putWriteBuffer(gradBuffer, true);
            gradient.putArg(gradBuffer);
            input.add(gradBuffer);
        }

        output = getBuffer((double[]) tensor.getGrad());
        queue.putWriteBuffer(output, true);
        gradient.putArg(output);

        for (Object param : params) {
            setObjectConvert(gradient, param);
        }

        gradient.rewind();
    }

    public void gradient(final Object... params) {
        setGradientArgs(params);
        queue.put2DRangeKernel(gradient, 0, 0, block.x, block.y, 0, 0);
        IntStream.range(0, input.size()).forEach(l -> {
            CLBuffer<DoubleBuffer> clBuffer = input.get(l);
            queue.putReadBuffer(clBuffer, true);
            DoubleBuffer buffer = clBuffer.getBuffer();
            Tensor in = tensor.getInput()[l];
            double[] value = (double[]) in.getValue();
            IntStream.range(0, value.length).forEach(i -> value[i] = buffer.get(i));
        });
        output.release();
    }

    public <T> CLBuffer getBuffer(double[] arr) {
        Buffer directBuffer = Buffers.newDirectDoubleBuffer(arr);
        CLBuffer<Buffer> buffer = context.createBuffer(directBuffer, READ_WRITE);
        buffer.getBuffer().position(0);
        return buffer;
    }

    public TensorContext setBlock(int... x) {
        if (x.length == 1) this.block = new Block(x[0]);
        if (x.length == 2) this.block = new Block(x[0], x[1]);
        if (x.length == 3) this.block = new Block(x[0], x[1], x[2]);
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
