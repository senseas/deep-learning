package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.annotation.Cuda;
import com.deep.framework.lang.cuda.Block;
import com.deep.framework.lang.cuda.Grid;
import com.deep.framework.lang.util.BeanUtil;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.nvrtc.JNvrtc;
import jcuda.nvrtc.nvrtcProgram;
import lombok.Data;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.nvrtc.JNvrtc.*;
import static jcuda.runtime.JCuda.cudaFree;

@Data
public class CudaExecutor implements Serializable {

    private static Map<String, CUfunction> functions = new HashMap<String, CUfunction>();
    private static Map<String, String> params = new HashMap<String, String>();
    private static Map<String, Boolean> parallels = new HashMap<String, Boolean>();

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

    @SneakyThrows
    public static void compute(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;

        CUfunction function = getFunction(tensor);
        String[] param = tensor.getOutParams().split(",");
        int length = param.length;

        double[] input = Arrays.stream(tensor.getInput()).flatMapToDouble(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                return Arrays.stream((double[]) a.getValue());
            } else {
                return Arrays.stream(new double[]{(double) a.getValue()});
            }
        }).toArray();

        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<None> tenser = tensor.getOutput();
            int size = tenser.size();
            if (tensor.isParallel()) {
                double[] output = new double[size * length];
                run(function, new Grid(size), new Block(1), input, output);
                tensor.setData(output);
                tensor.setValue(IntStream.range(0, size).mapToDouble(i -> output[i * length + length - 1]).toArray());
            } else {
                int l = length / size;
                double[] output = new double[length];
                run(function, new Grid(1), new Block(1), input, output);
                tensor.setData(output);
                tensor.setValue(IntStream.range(0, size).mapToDouble(i -> output[i * l + l - 1]).toArray());
            }
        } else {
            double[] output = new double[length];
            run(function, input, output);
            tensor.setData(output);
            None out = tensor.getOutput();
            out.setValue(output[length - 1]);
        }
    }

    @SneakyThrows
    public static void gradient(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;

        CUfunction function = getGradient(tensor);
        double[] input = Arrays.stream(tensor.getInput()).flatMapToDouble(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                return Arrays.stream((double[]) a.getValue());
            } else {
                return Arrays.stream(new double[]{(double) a.getValue()});
            }
        }).toArray();

        None[] list = Arrays.stream(tensor.getInput()).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).toArray(None[]::new);

        double[] inGrad = new double[list.length];
        if (BeanUtil.isTenser(tensor.getFunction())) {
            if (tensor.isParallel()) {
                int size = ((Tenser<None>) tensor.getOutput()).size();
                run(function, new Grid(size), new Block(1), input, tensor.getData(), tensor.getOutGradData(), inGrad);
                IntStream.range(0, list.length).forEach(i -> {
                    None none = list[i];
                    //none.setValue(inGrad[i]);
                });
            } else {
                run(function, input, tensor.getData(), tensor.getOutGradData(), inGrad);
                IntStream.range(0, list.length).forEach(i -> {
                    None none = list[i];
                    //none.setValue(inGrad[i]);
                });
            }
        } else {
            run(function, input, tensor.getData(), tensor.getOutGradData(), inGrad);
            IntStream.range(0, list.length).forEach(i -> {
                None none = list[i];
                //none.setValue(inGrad[i]);
            });
        }
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The operator
     * @return The CUDA function
     */
    public static CUfunction getFunction(Tensor tensor) {
        String name = tensor.getName().replace("Tensor::", "");

        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) {
            tensor.setOutParams(params.get(name));
            tensor.setParallel(parallels.get(name));
            return function;
        }

        isSame(tensor);

        TensorCore.inxMap = new HashMap<>();
        Arrays.stream(tensor.getInput()).forEach(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                output.forEach(out -> TensorCore.inxMap.put(out.getValId().trim(), TensorCore.inxMap.size()));
            } else {
                None out = a.getOutput();
                TensorCore.inxMap.put(out.getValId().trim(), TensorCore.inxMap.size());
            }
        });

        String code;
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (tensor.isParallel()) {
                TensorCore.func = TensorCore.code = TensorCore.inParams = TensorCore.outParams = "";
                TensorCore.forward(tenser.first());
                code = TensorCore.code.replace("compute", name);
            } else {
                TensorCore.func = TensorCore.code = TensorCore.inParams = TensorCore.outParams = "";
                tenser.forEach(TensorCore::forward);
                code = TensorCore.code.replace("compute", name);
            }
        } else {
            TensorCore.func = TensorCore.code = TensorCore.inParams = TensorCore.outParams = "";
            TensorCore.forward((Tensor) tensor.getFunction());
            code = TensorCore.code.replace("compute", name);
        }

        tensor.setOutParams(String.join(",", TensorCore.getParam(TensorCore.outParams)));
        tensor.setInParams(String.join(",", TensorCore.getParam(TensorCore.inParams)));
        System.out.println(code);
        function = createFunction(name, code);
        params.put(name, tensor.getOutParams());
        parallels.put(name, tensor.isParallel());
        functions.put(name, function);
        return function;
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The operator
     * @return The CUDA gradient function
     */
    public static CUfunction getGradient(Tensor tensor) {
        String name = tensor.getName().replace("Tensor::", "Grad");

        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) return function;

        TensorCore.inxMap = new HashMap<>();
        TensorCore.inxGradMap = new HashMap<>();
        Arrays.stream(tensor.getInput()).forEach(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                output.forEach(out -> {
                    int size = TensorCore.inxMap.size();
                    TensorCore.inxMap.put(out.getValId().trim(), size);
                    TensorCore.inxGradMap.put(out.getGradId().trim(), size);
                });
            } else {
                None out = a.getOutput();
                int size = TensorCore.inxMap.size();
                TensorCore.inxMap.put(out.getValId().trim(), size);
                TensorCore.inxGradMap.put(out.getGradId().trim(), size);
            }
        });

        String code;
        TensorCore.outGradParams = getGradOutParam(tensor);
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (tensor.isParallel()) {
                TensorCore.outParams = tensor.getOutParams();
                TensorCore.inParams = tensor.getInParams();
                TensorCore.grad = TensorCore.code = TensorCore.gradParams = "";
                TensorCore.backward(tenser.first());
                code = TensorCore.code.replace("gradient", name);
            } else {
                TensorCore.outParams = tensor.getOutParams();
                TensorCore.inParams = tensor.getInParams();
                TensorCore.grad = TensorCore.code = TensorCore.gradParams = "";
                tenser.forEach(TensorCore::backward);
                code = TensorCore.code.replace("gradient", name);
            }
        } else {
            TensorCore.outParams = tensor.getOutParams();
            TensorCore.inParams = tensor.getInParams();
            TensorCore.grad = TensorCore.code = TensorCore.gradParams = "";
            TensorCore.backward((Tensor) tensor.getFunction());
            code = TensorCore.code.replace("gradient", name);
        }

        System.out.println(code);
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
    }

    public static boolean isSame(Tensor tensor) {
        if (BeanUtil.isNotTenser(tensor.getFunction())) {
            tensor.setParallel(true);
            return true;
        }
        Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
        if (tenser.size() == 1) {
            tensor.setParallel(true);
            return true;
        }
        Tensor m = tenser.data(0), n = tenser.data(1);

        TensorCore.func = TensorCore.code = TensorCore.inParams = TensorCore.outParams = "";
        TensorCore.forward(m);
        String codem = TensorCore.code;

        TensorCore.func = TensorCore.code = TensorCore.inParams = TensorCore.outParams = "";
        TensorCore.forward(n);
        String coden = TensorCore.code;

        tensor.setParallel(codem.equals(coden));
        return tensor.isParallel();
    }

    public static String getGradOutParam(Tensor tensor) {
        if (BeanUtil.isTenser(tensor.getOutput())) {
            Tenser<None> output = tensor.getOutput();
            return output.stream().map(None::getGradId).collect(Collectors.joining(","));
        } else {
            None output = tensor.getOutput();
            return output.getGradId();
        }
    }

    public static void run(CUfunction function, Grid grid, Block block, double[] in, double[] out) {
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

    public static void run(CUfunction function, Grid grid, Block block, double[] in, double[] out, double[] outGrad, double[] inGrad) {
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