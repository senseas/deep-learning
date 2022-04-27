package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.cuda.Block;
import com.deep.framework.lang.cuda.Grid;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import lombok.Data;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.deep.framework.lang.ForEach.forEach;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;
import static jcuda.runtime.JCuda.cudaFree;

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

    public static void run(CUfunction function, Grid grid, Block block, double[] output) {
        CUdeviceptr outputDevice = createDeviceData(output);

        Pointer kernelParams = createKernelParams(outputDevice);
        cuLaunchKernel(function,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0, null,
            kernelParams,
            null
        );

        cuMemcpyDtoH(Pointer.to(output), outputDevice, output.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(outputDevice);
        cudaFree(kernelParams);
    }

    public static void run(CUfunction function, Grid grid, Block block, double[] input, double[] output) {
        CUdeviceptr inputDevice = createDeviceData(input);
        CUdeviceptr outputDevice = createDeviceData(output);

        Pointer kernelParams = createKernelParams(inputDevice, outputDevice);

        cuLaunchKernel(function,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0, null,
            kernelParams,
            null
        );

        cuMemcpyDtoH(Pointer.to(output), outputDevice, output.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(inputDevice);
        cudaFree(outputDevice);
        cudaFree(kernelParams);
    }

    public static void run(CUfunction function, double[] output) {
        CUdeviceptr outputDevice = createDeviceData(output);

        cuLaunchKernel(function,
            1, 1, 1,
            1, 1, 1,
            0, null,
            createKernelParams(outputDevice),
            null
        );

        cuMemcpyDtoH(Pointer.to(output), outputDevice, output.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(outputDevice);
    }

    public static void run(CUfunction function, double[] input, double[] output) {
        CUdeviceptr inputDevice = createDeviceData(input);
        CUdeviceptr outputDevice = createDeviceData(output);
        Pointer kernelParams = createKernelParams(inputDevice, outputDevice);

        cuLaunchKernel(function,
            1, 1, 1,
            1, 1, 1,
            0, null,
            kernelParams,
            null
        );

        cuMemcpyDtoH(Pointer.to(output), outputDevice, output.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(inputDevice);
        cudaFree(outputDevice);
        cudaFree(kernelParams);
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static void compute(Tensor tensor) {
        if (tensor.getFunction() instanceof Tenser) {
            Tenser<None> nones = TensorFlux.getOutput(tensor.getFunction());
            List<None> list = new ArrayList<>();
            nones.forEach((None a, int i) -> list.addAll(a.getParams()));
            CUfunction function = getFunction(tensor, nones.findFirst());
            double[] output = list.stream().mapToDouble(None::getValue).toArray();
            run(function, new Grid(nones.size()), new Block(1), output);
            nones.forEach((None a, int i) -> a.setValue(output[i]));
        } else {
            None out = ((Tensor) tensor.getFunction()).getOutput();
            CUfunction function = getFunction(tensor, out);
            double[] output = out.getParamx().stream().mapToDouble(None::getValue).toArray();
            run(function, output);
        }
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static void gradient(Tensor tensor) {
        if (tensor.getFunction() instanceof Tenser) {
            Tenser tenser = tensor.getOutput();
            IntStream.range(0, tensor.getInput().length).forEach(i -> {
                Tenser<None> nones = tensor.getInput()[i].getOutput();
                List<None> list = new ArrayList();
                forEach(nones, tenser, (None inx, None out) -> {
                    list.add(new None(out.getGrad()));
                    list.addAll(inx.getParams());
                });
                CUfunction function = getGradient(tensor, nones.findFirst(), i);
                double[] input = list.stream().mapToDouble(None::getValue).toArray();
                double[] output = new double[nones.size()];
                run(function, new Grid(nones.size()), new Block(1), input, output);
                nones.forEach((None inx, int l) -> inx.setGrad(output[l]));
            });
        } else {
            None out = tensor.getOutput();
            IntStream.range(0, tensor.getInput().length).forEach(i -> {
                None none = tensor.getInput()[i].getOutput();
                List<Double> list = new ArrayList<>();
                list.add(out.getGrad());
                none.getParams().forEach(b -> list.add(b.getValue()));
                CUfunction function = getGradient(tensor, none, 0);
                double[] input = list.stream().mapToDouble(Double::valueOf).toArray();
                double[] output = new double[1];
                run(function, input, output);
            });
        }
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static CUfunction getFunction(Tensor tensor, None none) {
        String name = tensor.getName().replace("Tensor::", "");
        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) return function;
        tensor.setCompCuda(true);
        String code = getFuncCode(name, none.getFuncs());
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static CUfunction getGradient(Tensor tensor, None none, int i) {
        String name = tensor.getName().replace("Tensor::", "").concat(String.valueOf(i));
        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) return function;
        tensor.setGradCuda(true);
        String code = getCode(name, none.getGrads());
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
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
     * Create device code
     *
     * @param name    of function
     * @param content The content of the code
     * @return The pointer to the data
     */
    public static String getFuncCode(String name, String content) {
        AtomicInteger index = new AtomicInteger();
        StringBuilder code = new StringBuilder("extern \"C\" __global__ void ");
        code.append(name).append("(double* out){ int idx = blockDim.x * blockIdx.x + threadIdx.x;");
        StringBuilder express = new StringBuilder();
        content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                express.append(a);
                return "{";
            }
            if (a.concat(b).equals("{var}")) {
                express.append("out[idx*M+").append(index).append("]");
                index.getAndIncrement();
                return "";
            }
            if (a.isEmpty()) {
                express.append(b);
                return "";
            }
            return a.concat(b);
        });
        code.append("int M = ").append(index.get()).append(";");
        code.append(express);
        return code.append(";}").toString();
    }

    /**
     * Create device code
     *
     * @param name    of function
     * @param content The content of the code
     * @return The pointer to the data
     */
    public static String getCode(String name, String content) {
        AtomicInteger index = new AtomicInteger();
        StringBuilder code = new StringBuilder("extern \"C\" __global__ void ");
        code.append(name).append("(double* inx , double* out){ int idx = blockDim.x * blockIdx.x + threadIdx.x;");
        StringBuilder express = new StringBuilder();
        content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                express.append(a);
                return "{";
            }
            if (a.concat(b).equals("{var}")) {
                express.append("inx[idx*M+").append(index).append("]");
                index.incrementAndGet();
                return "";
            }
            if (a.isEmpty()) {
                express.append(b);
                return "";
            }
            return a.concat(b);
        });
        code.append("int M = ").append(index.get()).append(";");
        code.append("out[idx] = ").append(express);
        return code.append(";}").toString().replaceAll("--", "");
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
