package com.deep.framework.lang;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLKernel;

import java.util.stream.Stream;

public class TensorContext {
    private final CLContext context;
    private final CLCommandQueue queue;
    private final CLKernel kernel;
    private Object[] values;

    public TensorContext(CLContext context, CLCommandQueue queue, CLKernel kernel) {
        this.context = context;
        this.kernel = kernel;
        this.queue = queue;
    }

    public void setArgs(final Object... values) {
        this.values = values;
        kernel.setArgs(Stream.of(values).map(a -> {
            if (a instanceof Tensor) {
                Tensor b = (Tensor) a;
                CLBuffer buffer = b.getBuffer(context);
                buffer.getBuffer().position(0);
                queue.putWriteBuffer(buffer, true);
                return buffer;
            }
            return a;
        }).toArray()).rewind();
    }

    public void excute(final long globalWorkSizeX, final long globalWorkSizeY) {
        queue.put2DRangeKernel(kernel, 0, 0, globalWorkSizeX, globalWorkSizeY, 0, 0);
        Stream.of(values).forEach(a -> {
            if (a instanceof Tensor) {
                Tensor b = (Tensor) a;
                CLBuffer buffer = b.getBuffer(context);
                buffer.getBuffer().position(0);
                queue.putReadBuffer(buffer, true);
            }
        });
    }

}