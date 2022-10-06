package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.annotation.Cuda;
import com.deep.framework.lang.cuda.Block;
import com.deep.framework.lang.cuda.Grid;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import lombok.Data;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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
    @SneakyThrows
    public static void compute(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;
        if (tensor.getFunction() instanceof Tenser) {
            Tenser<None> nones = TensorFlux.getOutput(tensor.getFunction());
            CUfunction function = getFunction(tensor, nones.findFirst());
            List<None> list = new ArrayList<>();
            nones.forEach(a -> list.addAll(a.getFuncx()));
            double[] output = list.stream().mapToDouble(None::getValue).toArray();
            run(function, new Grid(nones.size()), new Block(1), output);
            IntStream.range(0, list.size()).forEach(i -> list.get(i).setValue(output[i]));
        } else {
            None out = ((Tensor) tensor.getFunction()).getOutput();
            CUfunction function = getFunction(tensor, out);
            List<None> list = out.getFuncx();
            double[] output = list.stream().mapToDouble(None::getValue).toArray();
            run(function, output);
            IntStream.range(0, list.size()).forEach(i -> list.get(i).setValue(output[i]));
        }
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    @SneakyThrows
    public static void gradient(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;
        IntStream.range(0, tensor.getInput().length).forEach(i -> {
            Object out = tensor.getInput()[i].getOutput();
            if (out instanceof Tenser) {
                Tenser<None> nones = (Tenser<None>) out;
                CUfunction function = getGradient(tensor, nones.findFirst(), i);
                List<None> list = new ArrayList<>();
                nones.forEach(a -> list.addAll(a.getGradx()));
                double[] input = list.stream().mapToDouble(None::getValue).toArray();
                double[] output = new double[nones.size()];
                run(function, new Grid(nones.size()), new Block(1), input, output);
                nones.forEach((None inx, int l) -> inx.setGradi(output[l]));
            } else {
                None none = (None) out;
                CUfunction function = getGradient(tensor, none, i);
                double[] input = none.getGradx().stream().mapToDouble(None::getValue).toArray();
                double[] output = new double[1];
                run(function, input, output);
                none.setGradi(output[0]);
            }
        });
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

        Map<String, Integer> map = new HashMap<>();
        IntStream.range(0, none.getFuncx().size()).forEach(i -> {
            None o = none.getFuncx().get(i);
            map.put(o.getValId(), i);
        });

        String code = getFuncCode(name, none.getFunc(), map);
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
    }

    /**
     * Create device code
     *
     * @param name    of function
     * @param content The content of the code
     * @return The pointer to the data
     */
    public static String getFuncCode(String name, String content, Map<String, Integer> map) {
        StringBuilder code = new StringBuilder("extern \"C\" __global__ void ").append(name).append("(double* data){");
        code.append("int idx = blockDim.x * blockIdx.x + threadIdx.x;");
        code.append("int M = ").append(map.size()).append(";");
        content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                code.append(a);
                return "{";
            }
            Integer inx = map.get(a.concat(b));
            if (Objects.nonNull(inx)) {
                code.append("data[idx*M+").append(inx).append("]");
                return "";
            }
            if (a.isEmpty()) {
                code.append(b);
                return "";
            }
            return a.concat(b);
        });
        code.append("}");
        return code.toString();
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static CUfunction getGradient(Tensor tensor, None none, int l) {
        String name = tensor.getName().replace("Tensor::", "").concat(String.valueOf(l));

        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) return function;

        Map<String, Integer> map = new HashMap<>();
        IntStream.range(0, none.getGradx().size()).forEach(i -> {
            None o = none.getGradx().get(i);
            map.put(o.getValId(), i);
            map.put("{" + o.getGradId() + "}", i);
        });

        String code = getGradCode(name, none.getGradc(), none, map);
        System.out.println(code);
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
    }

    /**
     * Create device code
     *
     * @param name    of function
     * @param content The content of the code
     * @return The pointer to the data
     */
    public static String getGradCode(String name, String content, None none, Map<String, Integer> map) {
        StringBuilder code = new StringBuilder("extern \"C\" __global__ void ").append(name).append("(double* data , double* grad){");
        code.append("int idx = blockDim.x * blockIdx.x + threadIdx.x;");
        code.append("int M = ").append(none.getGradx().size()).append(";");
        String val = Arrays.stream(none.getParan().split(",")).distinct().collect(Collectors.joining(","));
        code.append("double " + val + ";");
        content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                code.append(a);
                return "{";
            }
            Integer inx = map.get(a.concat(b));
            if (Objects.nonNull(inx)) {
                code.append("data[idx*M+").append(inx).append("]");
                return "";
            }
            if (a.isEmpty()) {
                code.append(b);
                return "";
            }
            return a.concat(b);
        });
        code.append("}");
        return code.toString().replace(none.getGradId() + "=", "grad[idx]+=");
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