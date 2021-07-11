package com.deep.framework.lang;

import com.jogamp.opencl.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

public class TensorExecutor {
    private final CLContext context;

    private CLCommandQueue queue;
    private Map<String, CLKernel> kernels;

    public TensorExecutor() {
        context = CLContext.create();
        try (InputStream stream = new FileInputStream("D:\\GitHub\\deep-learning\\deep-framework\\src\\test\\java\\com\\deep\\framework\\lang\\kernel.cl")) {
            queue = context.getMaxFlopsDevice(CLDevice.Type.GPU).createCommandQueue();
            CLProgram program = context.createProgram(stream).build();
            kernels = program.createCLKernels();
        } catch (IOException e) {
            context.release();
        }
    }

    public TensorContext createContext(Tensor tensor) {
        CLKernel compute = kernels.get(tensor.name);
        CLKernel gradient = kernels.get(tensor.name.concat("Gradient"));
        return new TensorContext(tensor, context, queue, compute, gradient);
    }

    public CLCommandQueue getQueue() { return queue; }
}