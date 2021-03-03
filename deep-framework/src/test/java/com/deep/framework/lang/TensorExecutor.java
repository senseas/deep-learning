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
        try (InputStream stream = new FileInputStream( "D:\\GitHub\\deep-learning\\deep-framework\\src\\test\\java\\com\\deep\\framework\\lang\\kernel.cl")) {
            queue = context.getMaxFlopsDevice(CLDevice.Type.GPU).createCommandQueue();
            CLProgram program = context.createProgram(stream).build();
            kernels = program.createCLKernels();
        } catch (IOException e) {
            context.release();
        }
    }

    public TensorContext createContext(String name) {
        CLKernel clKernel = kernels.get(name);
        return new TensorContext(context, queue, clKernel);
    }

    public CLCommandQueue getQueue() { return queue; }
}