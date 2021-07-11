package com.deep.framework.lang;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;

import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

public class TensorContext {
    private final Tensor tensor;
    private final CLContext context;
    private final CLCommandQueue queue;
    private final CLKernel compute, gradient;
    private Object[] values;

    public TensorContext(Tensor tensor, CLContext context, CLCommandQueue queue, CLKernel compute, CLKernel gradient) {
        this.tensor = tensor;
        this.context = context;
        this.queue = queue;
        this.compute = compute;
        this.gradient = gradient;
    }

    public TensorContext setArgs(final Object... c) {
        this.values = c;
        Stream<Tensor> concat = Stream.concat(Arrays.stream(tensor.input), Stream.of(tensor));
        Stream params = Stream.concat(concat.map(tensor -> {
            CLBuffer buffer = tensor.buffer = getBuffer(tensor);
            queue.putWriteBuffer(buffer, true);
            return buffer;
        }), Arrays.stream(c));
        compute.setArgs(params.toArray()).rewind();
        return this;
    }

    public Object compute(final long globalWorkSizeX, final long globalWorkSizeY) {
        queue.put2DRangeKernel(compute, 0, 0, globalWorkSizeX, globalWorkSizeY, 0, 0);
        queue.putReadBuffer(tensor.buffer, true);
        FloatBuffer buffer = (FloatBuffer) tensor.buffer.getBuffer();
        IntStream.range(0, buffer.capacity()).forEach(i -> tensor.output[i] = buffer.get(i));
        return tensor.output;
    }

    private TensorContext setArgs() {
        Stream<Tensor> concat = Stream.concat(Arrays.stream(tensor.input), Arrays.stream(tensor.input));
        concat = Stream.concat(concat, Stream.of(tensor.input));
        Stream params = Stream.concat(concat.map(tensor -> {
            CLBuffer buffer = tensor.buffer = getBuffer(tensor);
            queue.putWriteBuffer(buffer, true);
            return buffer;
        }), Arrays.stream(values));
        gradient.setArgs(params.toArray()).rewind();
        return this;
    }

    public Object gradient(final long globalWorkSizeX, final long globalWorkSizeY) {
        this.setArgs();
        queue.put2DRangeKernel(gradient, 0, 0, globalWorkSizeX, globalWorkSizeY, 0, 0);
        Stream.of(tensor.input).forEach(tensor -> {
            queue.putReadBuffer(tensor.buffer, true);
            FloatBuffer buffer = (FloatBuffer) tensor.buffer.getBuffer();
            IntStream.range(0, buffer.capacity()).forEach(i -> tensor.output[i] = buffer.get(i));
        });
        return tensor.output;
    }

    public CLBuffer getBuffer(Tensor tensor) {
        Buffer directBuffer = Buffers.newDirectFloatBuffer(tensor.output);
        CLBuffer<Buffer> buffer = context.createBuffer(directBuffer, READ_WRITE);
        buffer.getBuffer().position(0);
        return buffer;
    }

}