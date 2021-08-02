package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Block;
import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;

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

    private void setComputeArgs(final Object... c) {

        bufferList = new ArrayList();

        compute.setArgs(Arrays.stream(c).map(a -> {

            if (bufferList.size() <= tensor.getInput().length) {

                CLBuffer buffer = getBuffer(linesValue(a));

                queue.putWriteBuffer(buffer, true);

                bufferList.add(buffer);

                return buffer;
            }

            return a;

        }).toArray()).rewind();

    }


    public void compute(final Object... c) {

        setComputeArgs(c);

        queue.put2DRangeKernel(compute, 0, 0, block.x, block.y, 0, 0);

        CLBuffer clBuffer = bufferList.get(tensor.getInput().length);

        queue.putReadBuffer(clBuffer, true);

        FloatBuffer buffer = (FloatBuffer) clBuffer.getBuffer();

        AtomicInteger index = new AtomicInteger();

        Object output = c[tensor.getInput().length];

        forEach(output, (None a) -> a.setValue(buffer.get(index.getAndIncrement())));

    }

    private void setGradientArgs(final Object... c) {

        bufferList = new ArrayList();

        Object[] input = Arrays.copyOfRange(c, 0, tensor.getInput().length);

        for (Object a : input) {

            CLBuffer buffer = getBuffer(linesValue(a));

            queue.putWriteBuffer(buffer, true);

            gradient.putArg(buffer);

        }

        for (Object a : input) {

            CLBuffer buffer = getBuffer(linesGrad(a));

            queue.putWriteBuffer(buffer, true);

            gradient.putArg(buffer);

            bufferList.add(buffer);

        }

        Object output = c[tensor.getInput().length];

        CLBuffer buffer = getBuffer(linesGrad(output));

        queue.putWriteBuffer(buffer, true);

        gradient.putArg(buffer);


        Object[] params = Arrays.copyOfRange(c, tensor.getInput().length + 1, c.length);

        for (Object a : params) {

            gradient.putArg((int) a);

        }

        gradient.rewind();

    }

    public void gradient(final Object... c) {

        setGradientArgs(c);

        queue.put2DRangeKernel(gradient, 0, 0, block.x, block.y, 0, 0);

        IntStream.range(0, tensor.getInput().length).forEach(i -> {

            Object input = c[i];

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
}