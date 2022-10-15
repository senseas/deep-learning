package com.deep.framework.framework;

import com.alibaba.fastjson.JSONObject;
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

    public static void run(CUfunction function, Grid grid, Block block, double[] data, double[] gradx, double[] grad) {
        CUdeviceptr dataDevice = createDeviceData(data);
        CUdeviceptr gradxDevice = createDeviceData(gradx);
        CUdeviceptr gradDevice = createDeviceData(grad);

        Pointer kernelParams = createKernelParams(dataDevice, gradxDevice, gradDevice);

        cuLaunchKernel(function,
            grid.x, grid.y, grid.z,
            block.x, block.y, block.z,
            0, null,
            kernelParams,
            null
        );

        cuMemcpyDtoH(Pointer.to(grad), gradDevice, grad.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(dataDevice);
        cudaFree(gradxDevice);
        cudaFree(gradDevice);
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

    public static void run(CUfunction function, double[] data, double[] gradx, double[] grad) {
        CUdeviceptr dataDevice = createDeviceData(data);
        CUdeviceptr gradxDevice = createDeviceData(gradx);
        CUdeviceptr gradDevice = createDeviceData(grad);

        Pointer kernelParams = createKernelParams(dataDevice, gradxDevice, gradDevice);

        cuLaunchKernel(function,
            1, 1, 1,
            1, 1, 1,
            0, null,
            kernelParams,
            null
        );

        cuMemcpyDtoH(Pointer.to(grad), gradDevice, grad.length * Sizeof.DOUBLE);
        cuCtxSynchronize();

        cudaFree(dataDevice);
        cudaFree(gradxDevice);
        cudaFree(gradDevice);
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

        CUfunction function = getFunction(tensor);
        String[] param = tensor.getFparam().split(",");
        int length = param.length;

        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<None> tenser = tensor.getOutput();
            Map<String, None> mapper = tenser.stream().collect(Collectors.toMap(a -> a.getValId().trim(), b -> b));

            int size = tenser.size();
            if (isSame(tensor)) {
                Map<String, Tensor> map = new HashMap<>();
                Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).forEach(a -> {
                    if (BeanUtil.isTenser(a.getOutput())) {
                        Tenser<None> output = a.getOutput();
                        map.put(output.first().getValId().trim(), a);
                    } else {
                        None out = a.getOutput();
                        map.put(out.getValId().trim(), a);
                    }
                });

                double[] output = new double[size * length];
                IntStream.range(0, size).forEach(i -> {
                    IntStream.range(0, length).forEach(l -> {
                        Tensor none = map.get(param[l]);
                        if (Objects.nonNull(none)) {
                            double[] values = (double[]) none.getValue();
                            output[i * length + l] = values[i];
                        }
                    });
                });

                run(function, new Grid(size), new Block(1), output);
                tensor.setData(output);
                IntStream.range(0, size).forEach(i -> {
                    None none = tenser.data(i);
                    none.setValue(output[i * length + length - 1]);
                });
            } else {
                Map<String, None> map = new HashMap<>();
                Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).forEach(a -> {
                    if (BeanUtil.isTenser(a.getOutput())) {
                        Tenser<None> output = a.getOutput();
                        output.forEach(out -> map.put(out.getValId().trim(), out));
                    } else {
                        None out = a.getOutput();
                        map.put(out.getValId().trim(), out);
                    }
                });

                double[] output = new double[length];
                IntStream.range(0, length).forEach(l -> {
                    None none = map.get(param[l]);
                    if (Objects.nonNull(none)) {
                        output[l] = none.getValue();
                    }
                });

                run(function, new Grid(size), new Block(1), output);
                tensor.setData(output);
                IntStream.range(0, length).forEach(l -> {
                    None none = mapper.get(param[l]);
                    if (Objects.nonNull(none)) {
                        none.setValue(output[l]);
                    }
                });
            }
        } else {
            Map<String, None> map = new HashMap<>();
            Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).forEach(a -> {
                if (BeanUtil.isTenser(a.getOutput())) {
                    Tenser<None> output = a.getOutput();
                    output.forEach(out -> map.put(out.getValId().trim(), out));
                } else {
                    None out = a.getOutput();
                    map.put(out.getValId().trim(), out);
                }
            });

            double[] output = new double[length];
            IntStream.range(0, length).forEach(l -> {
                None none = map.get(param[l]);
                if (Objects.nonNull(none)) {
                    output[l] = none.getValue();
                }
            });

            run(function, output);
            tensor.setData(output);
            None out = tensor.getOutput();
            out.setValue(output[length - 1]);
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

        CUfunction function = getGradient(tensor);
        None[] list = Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).toArray(None[]::new);

        double[] output = new double[list.length];
        if (BeanUtil.isTenser(tensor.getFunction())) {
            if (isSame(tensor)) {
                int size = ((Tenser<None>) tensor.getOutput()).size();
                run(function, new Grid(size), new Block(1), tensor.getData(), tensor.getGradOutData(), output);
                IntStream.range(0, list.length).forEach(i -> {
                    None none = list[i];
                    none.setValue(output[i]);
                });
            } else {
                run(function, tensor.getData(), tensor.getGradOutData(), output);
                IntStream.range(0, list.length).forEach(i -> {
                    None none = list[i];
                    none.setValue(output[i]);
                });
            }
        } else {
            run(function, tensor.getData(), tensor.getGradOutData(), output);
            IntStream.range(0, list.length).forEach(i -> {
                None none = list[i];
                none.setValue(output[i]);
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
    public static CUfunction getFunction(Tensor tensor) {
        String name = tensor.getName().replace("Tensor::", "");

        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) return function;

        String code;
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (isSame(tensor)) {
                TensorCore.func = TensorCore.code = TensorCore.fparam = "";
                TensorCore.forward(tenser.first());
                code = TensorCore.code.replace("compute", name);
            } else {
                TensorCore.func = TensorCore.code = TensorCore.fparam = "";
                tenser.forEach(TensorCore::forward);
                code = TensorCore.code.replace("compute", name);
            }
        } else {
            TensorCore.func = TensorCore.code = TensorCore.fparam = "";
            TensorCore.forward((Tensor) tensor.getFunction());
            code = TensorCore.code.replace("compute", name);
        }

        tensor.setFparam(String.join(",", TensorCore.getParam(TensorCore.fparam)));
        System.out.println(code);
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
    }

    public static Boolean isSame(Tensor tensor) {
        Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
        if (tenser.size() == 1) return true;
        Tensor m = tenser.data(0), n = tenser.data(1);

        TensorCore.func = TensorCore.code = TensorCore.fparam = "";
        TensorCore.forward(m);
        String codem = TensorCore.code;

        TensorCore.func = TensorCore.code = TensorCore.fparam = "";
        TensorCore.forward(n);
        String coden = TensorCore.code;
        return codem.equals(coden);
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static CUfunction getGradient(Tensor tensor) {
        String name = tensor.getName().replace("Tensor::", "Grad");

        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) return function;

        String code;
        TensorCore.next = getGradNextParam(tensor);
        TensorCore.gradout = getGradOutParam(tensor);
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (isSame(tensor)) {
                TensorCore.fparam = tensor.getFparam();
                TensorCore.grad = TensorCore.code = TensorCore.gparam = "";
                TensorCore.backward(tenser.first());
                code = TensorCore.code.replace("gradient", name);
            } else {
                TensorCore.fparam = tensor.getFparam();
                TensorCore.grad = TensorCore.code = TensorCore.gparam = "";
                tenser.forEach(TensorCore::backward);
                code = TensorCore.code.replace("gradient", name);
            }
        } else {
            TensorCore.fparam = tensor.getFparam();
            TensorCore.grad = TensorCore.code = TensorCore.gparam = "";
            TensorCore.backward((Tensor) tensor.getFunction());
            code = TensorCore.code.replace("gradient", name);
        }

        tensor.setGparam(String.join(",", TensorCore.getParam(TensorCore.gparam)));
        System.out.println(code);
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
    }

    public static String getGradNextParam(Tensor tensor) {
        List<Integer> nexts = new ArrayList(List.of(0));
        Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).forEach(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                nexts.add(nexts.get(nexts.size() - 1) + output.size());
            } else {
                nexts.add(nexts.get(nexts.size() - 1) + 1);
            }
        });
        return nexts.stream().map(String::valueOf).collect(Collectors.joining(","));
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