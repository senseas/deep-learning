package com.deep.framework.cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.nvrtc.nvrtcProgram;

import java.util.stream.Stream;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

public class Cuda {

    /**
     * Perform a default initialization of CUDA, creating a context
     * for the first device
     */
    static {
        /*JCudaDriver.setExceptionsEnabled(true);
        JNvrtc.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);*/
    }

    public static void run(CUfunction function, Dim grid, Dim block, double[] in, double[] out) {
        CUdeviceptr inDevice = createDeviceData(in);
        CUdeviceptr outDevice = createDeviceData(out);

        Pointer kernelParams = createKernelParams(inDevice, outDevice);
        cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, null, kernelParams, null);

        cuMemcpyDtoH(Pointer.to(out), outDevice, out.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(inDevice);
        cudaFree(outDevice);
        cudaFree(kernelParams);
    }

    public static void run(CUfunction function, double[] in, double[] out) {
        CUdeviceptr inDevice = createDeviceData(in);
        CUdeviceptr outDevice = createDeviceData(out);

        Pointer kernelParams = createKernelParams(inDevice, outDevice);
        cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, null, kernelParams, null);

        cuMemcpyDtoH(Pointer.to(out), outDevice, out.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(inDevice);
        cudaFree(outDevice);
        cudaFree(kernelParams);
    }

    public static void run(CUfunction function, Dim grid, Dim block, double[] in, double[] out, double[] outGrad, double[] inGrad) {
        CUdeviceptr inDevice = createDeviceData(in);
        CUdeviceptr outDevice = createDeviceData(out);
        CUdeviceptr outGradDevice = createDeviceData(outGrad);
        CUdeviceptr inGradDevice = createDeviceData(inGrad);

        Pointer kernelParams = createKernelParams(inDevice, outDevice, outGradDevice, inGradDevice);

        cuLaunchKernel(function, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, null, kernelParams, null);

        cuMemcpyDtoH(Pointer.to(inGrad), inGradDevice, inGrad.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(inDevice);
        cudaFree(outDevice);
        cudaFree(outGradDevice);
        cudaFree(inGradDevice);
        cudaFree(kernelParams);
    }

    public static void run(CUfunction function, double[] in, double[] out, double[] outGrad, double[] inGrad) {
        CUdeviceptr inDevice = createDeviceData(in);
        CUdeviceptr outDevice = createDeviceData(out);
        CUdeviceptr outGradDevice = createDeviceData(outGrad);
        CUdeviceptr inGradDevice = createDeviceData(inGrad);

        Pointer kernelParams = createKernelParams(inDevice, outDevice, outGradDevice, inGradDevice);

        cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, null, kernelParams, null);

        cuMemcpyDtoH(Pointer.to(inGrad), inGradDevice, inGrad.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(inDevice);
        cudaFree(outDevice);
        cudaFree(outGradDevice);
        cudaFree(inGradDevice);
        cudaFree(kernelParams);
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
        nvrtcCompileProgram(program, 1, new String[]{"--gpu-architecture=compute_61"});

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
        cudaMalloc(deviceData, value.length == 0 ? 1 : value.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceData, Pointer.to(value), value.length * Sizeof.DOUBLE);
        return deviceData;
    }

    /**
     * Create device data containing the given float value, the given number
     * of times
     *
     * @param data The value of the elements
     * @return The pointer to the data
     */
    public static Pointer createDevicePointer(double[] data) {
        int size = data.length * Sizeof.DOUBLE;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, size);
        cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice);
        return deviceData;
    }

    /**
     * copy host data to device
     * @param data The value of the elements
     * @return void
     */
    public static void copyDataHostToDevice(double[] data, Pointer deviceData) {
        int size = data.length * Sizeof.DOUBLE;
        cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice);
    }

    /**
     * copy device data to host
     * @param data The value of the elements
     * @return void
     */
    public static void copyDataDeviceToHost(double[] data, Pointer deviceData) {
        int size = data.length * Sizeof.DOUBLE;
        cudaMemcpy(Pointer.to(data), deviceData, size, cudaMemcpyDeviceToHost);
    }

    /**
     * Create device data containing the given float value, the given number
     * of times
     *
     * @param data The value of the elements
     * @return The pointer to the data
     */
    public static Pointer createDevicePointer(float[] data) {
        int size = data.length * Sizeof.FLOAT;
        Pointer deviceData = new Pointer();
        cudaMalloc(deviceData, size);
        cudaMemcpy(deviceData, Pointer.to(data), size, cudaMemcpyHostToDevice);
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