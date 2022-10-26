package com.deep.framework.framework;

import com.deep.framework.cuda.Dim;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.annotation.Cuda;
import com.deep.framework.lang.util.BeanUtil;
import jcuda.driver.CUfunction;
import lombok.Data;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.deep.framework.cuda.Cuda.createFunction;
import static com.deep.framework.cuda.Cuda.run;

@Data
public class CudaExecutor implements Serializable {

    private static Map<String, CUfunction> functions = new HashMap<>();
    private static Map<String, Tensor> parallels = new HashMap<>();
    private static TensorCore core;

    @SneakyThrows
    public static void compute(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;
        core = new TensorCore();
        CUfunction function = getFunction(tensor);
        double[] input = Arrays.stream(tensor.getInput()).flatMapToDouble(a -> Arrays.stream(a.getValue())).toArray();
        int length = tensor.getOutParams().size();

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
                tensor.setValue(IntStream.range(0, length).filter(i -> Objects.nonNull(map.get(tensor.getOutParams().get(i)))).mapToDouble(i -> output[i]).toArray());
            }
        } else {
            double[] output = new double[length];
            run(function, input, output);
            tensor.setData(output);
            None out = tensor.getOutput();
            out.setValue(output[length - 1]);
        }
    }

    public static void computes(Tensor tensor) {
        Tensor[] tensors = tensor.getInput();
        IntStream.range(0, tensor.getGrad().length).forEach(i -> tensor.getGrad()[i] = 0);
        Arrays.stream(tensors).flatMap(a -> Arrays.stream(a.getInput())).forEach(b -> ((None) b.getOutput()).setGradx(0.0));

        int size = tensors.length;
        core = new TensorCore(tensors[0].getInput().length);
        CUfunction function = getFunction(tensors[0]);
        tensor.setIparallel(true);

        double[] input = Arrays.stream(tensors).flatMap(a -> Arrays.stream(a.getInput()).map(b -> ((None) b.getOutput()).getValue())).mapToDouble(a -> a).toArray();
        int length = tensors[0].getOutParams().size();

        double[] output = new double[size * length];
        run(function, new Dim(size), new Dim(1), input, output);
        IntStream.range(0, size).forEach(i -> {
            Tensor in = tensors[i];
            in.setData(Arrays.copyOfRange(output, i * length, i * length + length));
            in.getValue()[0] = in.getData()[length - 1];
        });

        compute(tensor);
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

    @SneakyThrows
    public static void gradients(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;

        gradient(tensor);

        Tensor[] tensors = tensor.getInput();
        int size = tensors.length;
        core = new TensorCore(tensors[0].getInput().length);
        CUfunction function = getGradient(tensors[0]);

        double[] input = Arrays.stream(tensors).flatMap(a -> Arrays.stream(a.getInput()).map(b -> ((None) b.getOutput()).getValue())).mapToDouble(a -> a).toArray();
        double[] inGrad = new double[input.length];
        double[] output = Arrays.stream(tensors).flatMapToDouble(a -> Arrays.stream(a.getData())).toArray();
        double[] outGrad = Arrays.stream(tensors).flatMapToDouble(a -> Arrays.stream(a.getGrad())).toArray();

        run(function, new Dim(size), new Dim(1), input, output, outGrad, inGrad);
        IntStream.range(0, size).forEach(i -> {
            Tensor[] in = tensors[i].getInput();
            IntStream.range(0, in.length).forEach(l -> {
                None none = in[l].getOutput();
                none.setGradx(inGrad[i * in.length + l]);
            });
        });
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

        core.inxMap = new HashMap<>();
        Arrays.stream(tensor.getInput()).forEach(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                output.forEach(out -> core.inxMap.put(out.getValId().trim(), core.inxMap.size()));
            } else {
                None out = a.getOutput();
                core.inxMap.put(out.getValId().trim(), core.inxMap.size());
            }
        });

        String code;
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (tensor.isParallel()) {
                core.forward(tenser.first());
                code = core.code.replace("compute", name);
            } else {
                tenser.forEach(core::forward);
                code = core.code.replace("compute", name);
            }
        } else if (Objects.nonNull(tensor.getFunction())) {
            core.forward((Tensor) tensor.getFunction());
            code = core.code.replace("compute", name);
        } else {
            core.forward(tensor);
            code = core.code.replace("compute", name);
        }

        tensor.setOutParams(core.outParams);
        tensor.setInParams(core.inParams);
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

        core.outGradParams = getGradOutParam(tensor);
        core.inxGradMap = new HashMap<>();
        Arrays.stream(tensor.getInput()).forEach(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                output.forEach(out -> core.inxGradMap.put(out.getGradId().trim(), core.inxGradMap.size()));
            } else {
                None out = a.getOutput();
                core.inxGradMap.put(out.getGradId().trim(), core.inxGradMap.size());
            }
        });

        String code;
        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
            if (tensor.isParallel()) {
                core.backward(tenser.first());
                code = core.code.replace("gradient", name);
            } else {
                tenser.forEach(core::backward);
                code = core.code.replace("gradient", name);
            }
        } else if (Objects.nonNull(tensor.getFunction())) {
            core.backward((Tensor) tensor.getFunction());
            code = core.code.replace("gradient", name);
        } else {
            core.backward(tensor);
            code = core.code.replace("gradient", name);
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
                Tensor m = tenser.data(0), n = tenser.data(1);

                TensorCore corem = new TensorCore();
                corem.forward(m);

                TensorCore coren = new TensorCore();
                coren.forward(n);

                tensor.setParallel(corem.code.equals(coren.code));
            }
        }
        return tensor.isParallel();
    }

    public static boolean isSame(Tensor[] tensor) {
        if (!Objects.equals(tensor.length, 1)) {
            Tensor m = tensor[0], n = tensor[1];
            TensorCore corem = new TensorCore();
            corem.forward(m);

            TensorCore coren = new TensorCore();
            coren.forward(n);
            return corem.code.equals(coren.code);
        }
        return false;
    }

    public static List<String> getGradOutParam(Tensor tensor) {
        if (BeanUtil.isTenser(tensor.getOutput())) {
            Tenser<None> output = tensor.getOutput();
            return output.stream().map(None::getGradId).map(String::trim).toList();
        } else {
            None output = tensor.getOutput();
            return List.of(output.getGradId().trim());
        }
    }

}