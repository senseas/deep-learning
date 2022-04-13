package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.cuda.Block;
import com.deep.framework.lang.cuda.Grid;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;

@Data
public class CudaExecutor<E> implements Serializable {
    private static CudaExecutor executor;


    public CudaContext createContext(Tensor tensor) {
        initialize();
        return new CudaContext(tensor);
    }

    public static CudaExecutor New() {
        if (Objects.nonNull(executor)) return executor;
        return executor = new CudaExecutor();
    }

    public static void run(CUfunction function, Pointer parameters, Grid grid, Block block) {
        cuLaunchKernel(function,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0,
            null,
            parameters,
            null
        );
        cuCtxSynchronize();
    }

    /**
     * Perform a default initialization of CUDA, creating a context
     * for the first device
     */
    public static void initialize() {
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param name The name of the function
     * @param code The source code
     * @return The CUDA function
     */
    public static CUfunction createFunction(String name, String code) {
        nvrtcProgram program = new nvrtcProgram();
        nvrtcCreateProgram(program, code, null, 0, null, null);
        nvrtcCompileProgram(program, 0, null);

        String[] programLog = new String[1];
        nvrtcGetProgramLog(program, programLog);
        String log = programLog[0].trim();
        if (!log.isEmpty()) System.err.println("Compilation log for " + name + ":\n" + log);

        String[] ptx = new String[1];
        nvrtcGetPTX(program, ptx);
        nvrtcDestroyProgram(program);
        CUmodule module = new CUmodule();
        cuModuleLoadData(module, ptx[0]);
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, name);
        return function;
    }

    /**
     * Create device data containing the given float value, the given number
     * of times
     *
     * @param value The value of the elements
     * @return The pointer to the data
     */
    public static CUdeviceptr createDeviceData(double[] value) {
        CUdeviceptr deviceData = new CUdeviceptr();
        cuMemAlloc(deviceData, value.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceData, Pointer.to(value), value.length * Sizeof.DOUBLE);
        return deviceData;
    }

    /**
     * Create a pointer to the given kernel parameters. Note that only
     * a limited subset of argument types is supported here.
     *
     * @param args The kernel parameters
     * @return The pointer with the kernel parameters
     */
    public static Pointer createKernelParams(Object... args) {
        Pointer[] kernelParameters = new Pointer[args.length];
        for (int i = 0; i < args.length; i++) {
            Object arg = args[i];
            if (arg instanceof Pointer) {
                Pointer argPointer = (Pointer) arg;
                Pointer pointer = Pointer.to(argPointer);
                kernelParameters[i] = pointer;
            } else if (arg instanceof Integer) {
                Integer value = (Integer) arg;
                Pointer pointer = Pointer.to(new int[]{value});
                kernelParameters[i] = pointer;
            } else if (arg instanceof Float) {
                Float value = (Float) arg;
                Pointer pointer = Pointer.to(new float[]{value});
                kernelParameters[i] = pointer;
            } else {
                System.out.println("Type not supported: " + arg.getClass());
            }
        }
        return Pointer.to(kernelParameters);
    }
}
