package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.util.FileUtil;
import com.jogamp.opencl.*;
import lombok.Data;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.Map;
import java.util.Objects;

@Data
public class TensorGpuExecutor<E> implements Serializable {
    private CLContext context;
    private CLCommandQueue queue;
    private Map<String, CLKernel> kernels;
    private static TensorGpuExecutor executor;

    private TensorGpuExecutor() {
        context = CLContext.create();
        try (InputStream stream = FileUtil.readResourceAsStream("kernel.cl")) {
            queue = context.getMaxFlopsDevice(CLDevice.Type.GPU).createCommandQueue();
            CLProgram program = context.createProgram(stream).build();
            kernels = program.createCLKernels();
        } catch (IOException e) {
            context.release();
        }
    }

    public TensorContext createContext(Tensor tensor) {
        CLKernel compute = kernels.get(tensor.getName().replace("Tensor::",""));
        CLKernel gradient = kernels.get(tensor.getName().replace("Tensor::","").concat("Gradient"));
        return new TensorContext(tensor, context, queue, compute, gradient);
    }

    public CLCommandQueue getQueue() { return queue; }

    public static TensorGpuExecutor New() {
        if (Objects.nonNull(executor)) return executor;
        return new TensorGpuExecutor();
    }

}
