package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Block;
import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.*;

import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.forEach;
import static com.deep.framework.lang.Shape.linesGrad;
import static com.deep.framework.lang.Shape.linesValue;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

public class TensorContext {

    private final Tensor tensor;

    private final CLContext context;

    private final CLCommandQueue queue;

    private final CLKernel compute, gradient;

    private List<CLBuffer> bufferList;

    private Block block;

    public TensorContext(Tensor tensor, CLContext context, CLCommandQueue queue, CLKernel compute, CLKernel gradient) {

        this.tensor = tensor;

        this.context = context;

        this.queue = queue;

        this.compute = compute;

        this.gradient = gradient;
    }

    private void setComputeArgs(final Object... params) {

        bufferList = new ArrayList();

        compute.setArgs(Arrays.stream(params).map(a -> {

            if (bufferList.size() <= tensor.getInput().length) {

                CLBuffer buffer = getBuffer(linesValue(a));

                queue.putWriteBuffer(buffer, true);

                bufferList.add(buffer);

                return buffer;

            }

            return a;

        }).toArray()).rewind();

    }


    public void compute(final Object... params) {

        setComputeArgs(params);

        queue.put2DRangeKernel(compute, 0, 0, block.x, block.y, 0, 0);

        Object output = params[tensor.getInput().length];

        CLBuffer clBuffer = bufferList.get(tensor.getInput().length);

        queue.putReadBuffer(clBuffer, true);

        FloatBuffer buffer = (FloatBuffer) clBuffer.getBuffer();

        AtomicInteger index = new AtomicInteger();

        forEach(output, (None a) -> a.setValue(buffer.get(index.getAndIncrement())));

    }

    private void setGradientArgs(final Object... params) {

        bufferList = new ArrayList();

        IntStream.range(0, tensor.getInput().length).forEach(i -> {

            CLBuffer buffer = getBuffer(linesValue(params[i]));

            queue.putWriteBuffer(buffer, true);

            gradient.putArg(buffer);

        });

        IntStream.range(0, tensor.getInput().length).forEach(i -> {

            CLBuffer buffer = getBuffer(linesGrad(params[i]));

            queue.putWriteBuffer(buffer, true);

            gradient.putArg(buffer);

            bufferList.add(buffer);

        });

        IntStream.range(tensor.getInput().length, params.length).forEach(i -> {

            if (i == tensor.getInput().length) {

                CLBuffer buffer = getBuffer(linesGrad(params[i]));

                queue.putWriteBuffer(buffer, true);

                gradient.putArg(buffer);

            } else {

                setObjectConvert(gradient, params[i]);

            }

        });

        gradient.rewind();

    }

    public void gradient(final Object... params) {

        setGradientArgs(params);

        queue.put2DRangeKernel(gradient, 0, 0, block.x, block.y, 0, 0);

        IntStream.range(0, tensor.getInput().length).forEach(i -> {

            Object input = params[i];

            CLBuffer clBuffer = bufferList.get(i);

            queue.putReadBuffer(clBuffer, true);

            FloatBuffer buffer = (FloatBuffer) clBuffer.getBuffer();

            AtomicInteger index = new AtomicInteger();

            forEach(input, (None a) -> a.setGrad(buffer.get(index.getAndIncrement())));

        });

    }

    public <T> CLBuffer getBuffer(float[] arr) {

        Buffer directBuffer = Buffers.newDirectFloatBuffer(arr);

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
