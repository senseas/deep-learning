package com.deep.framework.framework;

import com.deep.framework.graph.None;
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
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;

@Data
public class CudaExecutor implements Serializable {

    private static Map<String, CUfunction> functions = new HashMap<String, CUfunction>();

    /**
     * Perform a default initialization of CUDA, creating a context
     * for the first device
     */
    static {
        JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);
    }

    public static void run(CUfunction function, Grid grid, Block block, double[] input, double[] output) {
        CUdeviceptr inputDevice = createDeviceData(input);
        CUdeviceptr outputDevice = createDeviceData(output);

        cuLaunchKernel(function,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0, null,
            createKernelParams(inputDevice, outputDevice),
            null
        );

        cuMemcpyDtoH(Pointer.to(output), outputDevice, output.length * Sizeof.DOUBLE);
        cuCtxSynchronize();
    }

    public static void run(CUfunction function, double[] input, double[] output) {
        CUdeviceptr inputDevice = createDeviceData(input);
        CUdeviceptr outputDevice = createDeviceData(output);

        cuLaunchKernel(function,
            1, 1, 1,
            1, 1, 1,
            0, null,
            createKernelParams(inputDevice, outputDevice),
            null
        );

        cuMemcpyDtoH(Pointer.to(output), outputDevice, output.length * Sizeof.DOUBLE);
        cuCtxSynchronize();
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static void createFunction(Tensor tensor) {
        IntStream.range(0, tensor.getInput().length).forEach(i -> {
            None none = tensor.getInput()[i].getOutput();
            String name = tensor.getName().replace("Tensor::", "").concat(String.valueOf(i));
            CUfunction function = createFunction(name, none.getFunc(name));
            functions.put(name, function);
        });
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static CUfunction getFunction(Tensor tensor, int i) {
        String name = tensor.getName().replace("Tensor::", "").concat(String.valueOf(i));
        CUfunction cUfunction = functions.get(name);
        if (Objects.isNull(cUfunction)) {
            createFunction(tensor);
            cUfunction = functions.get(name);
        }
        return cUfunction;
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
        return Pointer.to(Stream.of(args).map(a -> {
            if (a instanceof Pointer) {
                Pointer pointer = (Pointer) a;
                return Pointer.to(pointer);
            } else if (a instanceof Integer) {
                Integer value = (Integer) a;
                return Pointer.to(new int[]{value});
            } else if (a instanceof Float) {
                Float value = (Float) a;
                return Pointer.to(new float[]{value});
            } else if (a instanceof Long) {
                Long value = (Long) a;
                return Pointer.to(new long[]{value});
            } else if (a instanceof Double) {
                Float value = (Float) a;
                return Pointer.to(new double[]{value});
            }
            return a;
        }).toArray(Pointer[]::new));
    }

    /**
     * Create a pointer to the given kernel parameters. Note that only
     * a limited subset of argument types is supported here.
     *
     * @param args The kernel parameters
     * @return The pointer with the kernel parameters
     */
    public static Object[] createParams(Object... args) {
        return Stream.of(args).map(a -> {
            if (a instanceof double[]) {
                return createDeviceData((double[]) a);
            }
            return a;
        }).toArray();
    }
}
