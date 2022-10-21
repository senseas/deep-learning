package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.annotation.Cuda;
import com.deep.framework.cuda.Dim;
import com.deep.framework.lang.util.BeanUtil;
import jcuda.driver.CUfunction;
import lombok.Data;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.deep.framework.cuda.Cuda.createFunction;
import static com.deep.framework.cuda.Cuda.run;

@Data
public class CudaExecutor implements Serializable {

    private static Map<String, CUfunction> functions = new HashMap<>();
    private static Map<String, Tensor> parallels = new HashMap<>();

    @SneakyThrows
    public static void compute(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;

        CUfunction function = getFunction(tensor);
        double[] input = Arrays.stream(tensor.getInput()).flatMapToDouble(a -> Arrays.stream(a.getValue())).toArray();
        String[] param = tensor.getOutParams().split(",");
        int length = param.length;

        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<None> tenser = tensor.getOutput();
            Map<String, None> map = tenser.stream().collect(Collectors.toMap(a -> a.getValId().trim(), a -> a));
            int size = tenser.size();
            if (tensor.isParallel()) {
                double[] output = new double[size * length];
                run(function, new Dim(size), new Dim(1), input, output);
                tensor.setData(output);
                tensor.setValue(IntStream.range(0, size).mapToDouble(i -> output[i * length + length - 1]).toArray());
            } else {
                double[] output = new double[length];
                run(function, new Dim(1), new Dim(1), input, output);
                tensor.setData(output);
                tensor.setValue(IntStream.range(0, length).filter(i -> Objects.nonNull(map.get(param[i]))).mapToDouble(i -> output[i]).toArray());
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
        double[] input = Arrays.stream(tensor.getInput()).flatMapToDouble(a -> Arrays.stream(a.getValue())).toArray();
        double[] inGrad = new double[input.length];
        int length = tensor.getInput().length, l = input.length / length;

        if (BeanUtil.isTenser(tensor.getFunction())) {
            if (tensor.isParallel()) {
                int size = ((Tenser<None>) tensor.getOutput()).size();
                run(function, new Dim(size), new Dim(1), input, tensor.getData(), tensor.getGrad(), inGrad);
                IntStream.range(0, length).forEach(i -> {
                    int from = i * l;
                    Tensor in = tensor.getInput()[i];
                    in.setGrad(Arrays.copyOfRange(inGrad, from, from + l));
                });
            } else {
                run(function, input, tensor.getData(), tensor.getGrad(), inGrad);
                IntStream.range(0, length).forEach(i -> {
                    int from = i * l;
                    Tensor in = tensor.getInput()[i];
                    in.setGrad(Arrays.copyOfRange(inGrad, from, from + l));
                });
            }
        } else {
            run(function, input, tensor.getData(), tensor.getGrad(), inGrad);
            IntStream.range(0, length).forEach(i -> {
                int from = i * l;
                Tensor in = tensor.getInput()[i];
                in.setGrad(Arrays.copyOfRange(inGrad, from, from + l));
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
            Tensor parallel = parallels.get(name);
            tensor.setOutParams(parallel.getOutParams());
            tensor.setParallel(parallel.isParallel());
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
                TensorCore.forwardClear();
                TensorCore.forward(tenser.first());
                code = TensorCore.code.replace("compute", name);
            } else {
                TensorCore.forwardClear();
                tenser.forEach(TensorCore::forward);
                code = TensorCore.code.replace("compute", name);
            }
        } else {
            TensorCore.forwardClear();
            TensorCore.forward((Tensor) tensor.getFunction());
            code = TensorCore.code.replace("compute", name);
        }

        tensor.setOutParams(String.join(",", TensorCore.getParam(TensorCore.outParams)));
        tensor.setInParams(String.join(",", TensorCore.getParam(TensorCore.inParams)));
        System.out.println(code);
        function = createFunction(name, code);
        parallels.put(name, tensor);
        functions.put(name, function);

        getGradient(tensor);

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

        TensorCore.inxGradMap = new HashMap<>();
        Arrays.stream(tensor.getInput()).forEach(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                output.forEach(out -> TensorCore.inxGradMap.put(out.getGradId().trim(), TensorCore.inxGradMap.size()));
            } else {
                None out = a.getOutput();
                TensorCore.inxGradMap.put(out.getGradId().trim(), TensorCore.inxGradMap.size());
            }
        });

        String code;
        TensorCore.outGradParams = getGradOutParam(tensor);
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (tensor.isParallel()) {
                TensorCore.backwardClear();
                TensorCore.backward(tenser.first());
                code = TensorCore.code.replace("gradient", name);
            } else {
                TensorCore.backwardClear();
                tenser.forEach(TensorCore::backward);
                code = TensorCore.code.replace("gradient", name);
            }
        } else {
            TensorCore.backwardClear();
            TensorCore.backward((Tensor) tensor.getFunction());
            code = TensorCore.code.replace("gradient", name);
        }

        System.out.println(code);
        function = createFunction(name, code);
        functions.put(name, function);
        return function;
    }

    public static boolean isSame(Tensor tensor) {
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (!Objects.equals(tenser.size(), 1)) {
                TensorCore.inxMap = null;
                Tensor m = tenser.data(0), n = tenser.data(1);

                TensorCore.forwardClear();
                TensorCore.forward(m);
                String codem = TensorCore.code;

                TensorCore.forwardClear();
                TensorCore.forward(n);
                String coden = TensorCore.code;

                tensor.setParallel(codem.equals(coden));
            }
        }
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

}