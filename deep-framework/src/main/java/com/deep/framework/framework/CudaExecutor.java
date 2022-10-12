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

        CUfunction function = getFunction(tensor);
        String[] param = TensorCore.fparam.split(",");
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
        /*if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;
        IntStream.range(0, tensor.getInput().length).forEach(i -> {
            Object out = tensor.getInput()[i].getOutput();
            if (out instanceof Tenser) {
                Tenser<None> nones = (Tenser<None>) out;
                CUfunction function = getGradient(tensor, nones.first(), i);
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
        });*/
    }

    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    @SneakyThrows
    public static void gradientx(Tensor tensor) {
        /*if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;
        Object out = tensor.getInput()[0].getOutput();
        List<None> list = new ArrayList<>();
        int length = tensor.getInput().length;
        if (out instanceof Tenser) {
            int size = ((Tenser<None>) out).size();

            IntStream.range(0, size).forEach(i -> {
                list.addAll(IntStream.range(0, length).mapToObj(l -> {
                    Tenser<None> nones = tensor.getInput()[l].getOutput();
                    None none = nones.data(i);
                    return none.getGradx().stream();
                }).flatMap(a -> a).distinct().toList());
            });

            double[] input = list.stream().mapToDouble(None::getValue).toArray();
            double[] output = new double[length * size];
            CUfunction function = getGradientx(tensor);
            run(function, new Grid(size), new Block(1), input, output);

            IntStream.range(0, size).forEach(i -> {
                IntStream.range(0, length).forEach(l -> {
                    Tenser<None> nones = tensor.getInput()[l].getOutput();
                    None none = nones.data(i);
                    none.setGradi(output[i * length + l]);
                });
            });

        } else {

            IntStream.range(0, length).forEach(l -> {
                None none = tensor.getInput()[l].getOutput();
                list.addAll(none.getGradx());
            });

            double[] input = list.stream().distinct().mapToDouble(None::getValue).toArray();
            double[] output = new double[length];
            CUfunction function = getGradientx(tensor);
            run(function, new Grid(length), new Block(1), input, output);

            IntStream.range(0, length).forEach(l -> {
                None none = tensor.getInput()[l].getOutput();
                none.setGradi(output[l]);
            });
        }*/
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
        code.append(content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            Integer inx = map.get(a);
            if (Objects.nonNull(inx)) {
                code.append("data[idx*M+").append(inx).append("]");
                return b;
            }
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                code.append(a);
                return "{";
            }
            return a.concat(b);
        }).get());
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

        /*Map<String, Integer> map = new HashMap<>();
        IntStream.range(0, none.getGradx().size()).forEach(i -> {
            None o = none.getGradx().get(i);
            map.put(o.getValId(), i);
            map.put("{" + o.getGradId() + "}", i);
        });

        String code = getGradCode(name, none.getGradc(), none, map);
        System.out.println(code);
        function = createFunction(name, code);
        functions.put(name, function);*/

        return function;
    }
    /**
     * Create a CUDA kernel function by compiling the given code using the
     * NVRTC, and obtaining the function with the given name
     *
     * @param tensor The source code
     * @return The CUDA function
     */
    public static CUfunction getGradientx(Tensor tensor) {
        String name = tensor.getName().replace("Tensor::", "Grad");
        CUfunction function = functions.get(name);
        if (Objects.nonNull(function)) return function;

        /*StringBuffer para = new StringBuffer();
        StringBuffer code = new StringBuffer();
        List<None> paraList = new ArrayList<>();
        List<None> gradList = new ArrayList<>();

        Arrays.stream(tensor.getInput()).map(in -> {
            Object out = in.getOutput();
            if (out instanceof Tenser) {
                Tenser<None> nones = (Tenser<None>) out;
                return nones.first();
            } else {
                None none = (None) out;
                return none;
            }
        }).forEach((a) -> {
            para.append(a.getParan());
            code.append(a.getGradc());
            paraList.addAll(a.getGradx());
            gradList.add(a);
        });

        List<None> paraLists = paraList.stream().distinct().toList();
        Map<String, Integer> data = new HashMap<>();
        IntStream.range(0, paraLists.size()).forEach(i -> {
            None o = paraLists.get(i);
            data.put(o.getValId(), i);
            data.put("{" + o.getGradId() + "}", i);
        });

        Map<String, Integer> grad = new HashMap<>();
        IntStream.range(0, gradList.size()).forEach(i -> {
            None o = gradList.get(i);
            grad.put(o.getGradId().concat("="), i);
        });

        String gradCode = getGradCode(name, gradList.size(), paraLists.size(), para.toString(), code.toString(), grad, data);

        System.out.println(gradCode);
        function = createFunction(name, gradCode);
        functions.put(name, function);*/

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
       /* code.append("int idx = blockDim.x * blockIdx.x + threadIdx.x;");
        code.append("int M = ").append(none.getGradx().size()).append(";");
        String val = Arrays.stream(none.getParan().split(",")).distinct().collect(Collectors.joining(","));
        code.append("double " + val + ";");
        code.append(content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            Integer inx = map.get(a);
            if (Objects.nonNull(inx)) {
                code.append("data[idx*M+").append(inx).append("]");
                return b;
            }
            if (Objects.equals(none.getGradId().concat("="), a)) {
                code.append("grad[idx]+=");
                return b;
            }
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                code.append(a);
                return "{";
            }
            if (a.equals(";")) {
                code.append(a);
                return b;
            }
            if (b.equals(";")) {
                code.append(a);
                return ";";
            }
            return a.concat(b);
        }).get());
        code.append("}");*/
        return code.toString();
    }

    /**
     * Create device code
     *
     * @param name    of function
     * @param content The content of the code
     * @return The pointer to the data
     */
    public static String getGradCode(String name, Integer sizes, Integer size, String var, String content, Map<String, Integer> grad, Map<String, Integer> param) {
        var = Arrays.stream(var.split(",")).distinct().collect(Collectors.joining(",")).concat(";");
        content = Arrays.stream(content.split(";")).distinct().collect(Collectors.joining(";")).concat(";");

        StringBuilder code = new StringBuilder("extern \"C\" __global__ void ").append(name).append("(double* data , double* grad){");
        code.append("int idx = blockDim.x * blockIdx.x + threadIdx.x;");
        code.append("int M = ").append(size).append(";");
        code.append("int N = ").append(sizes).append(";");
        code.append("double ").append(var);

        code.append(content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            Integer inx = param.get(a);
            if (Objects.nonNull(inx)) {
                code.append("data[idx*M+").append(inx).append("]");
                return b;
            }
            Integer ing = grad.get(a);
            if (Objects.nonNull(ing)) {
                code.append("grad[idx*N+").append(ing).append("]+=");
                return b;
            }
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                code.append(a);
                return "{";
            }
            if (a.equals(";")) {
                code.append(a);
                return b;
            }
            if (b.equals(";")) {
                code.append(a);
                return ";";
            }
            return a.concat(b);
        }).get());
        code.append("}");
        return code.toString();
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