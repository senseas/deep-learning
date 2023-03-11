package com.deep.framework.cuda;

import com.deep.framework.creater.ParamCreater;
import com.deep.framework.creater.CudaCreater;
import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFunction;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.annotation.Cuda;
import com.deep.framework.lang.util.BeanUtil;
import jcuda.driver.CUfunction;
import lombok.Data;
import lombok.SneakyThrows;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.IntStream;

import static com.deep.framework.cuda.Cuda.createFunction;
import static com.deep.framework.cuda.Cuda.run;

@Data
public class CudaExecutor implements Serializable {

    private static Map<String, CUfunction> functions = new HashMap<>();
    private static ParamCreater creater;

    @SneakyThrows
    public static void compute(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;

        creater = new ParamCreater();
        CUfunction function = getFunction(tensor);
        double[] input = tensor.getCore().inxParams.stream().mapToDouble(None::getValue).toArray();
        double[] output = new double[tensor.getCore().outxParams.size()];

        if (BeanUtil.isTenser(tensor.getFunction())) {
            Tenser<None> tenser = tensor.getOutput();
            int size = tenser.size();
            run(function, new Dim(size), new Dim(1), input, output);
        } else {
            run(function, input, output);
        }
    }


    @SneakyThrows
    public static void gradient(Tensor tensor) {
        if (!tensor.getClass().getMethod("compute").isAnnotationPresent(Cuda.class)) return;

        CUfunction function = getGradient(tensor);
        double[] input = tensor.getCore().inParams.stream().mapToDouble(None::getValue).toArray();
        double[] output = tensor.getCore().outParams.stream().mapToDouble(None::getValue).toArray();
        double[] outGrad = tensor.getCore().outGradParams.stream().mapToDouble(None::getGrad).toArray();
        double[] inGrad = new double[tensor.getCore().inGradParams.size()];

        if (BeanUtil.isTenser(tensor.getFunction())) {
            int size = ((Tenser<None>) tensor.getOutput()).size();
            run(function, new Dim(size), new Dim(1), input, output, outGrad, inGrad);
            IntStream.range(0, inGrad.length).forEach(i -> tensor.getCore().inGradParams.get(i).setGrad(inGrad[i]));
        } else {
            run(function, input, output, tensor.getGrad(), inGrad);
            IntStream.range(0, inGrad.length).forEach(i -> tensor.getCore().inGradParams.get(i).setGrad(inGrad[i]));
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
        if (Objects.nonNull(function)) return function;

        CudaCreater cudac = new CudaCreater();
        if (tensor instanceof TensorFunction) {
            if (BeanUtil.isTenser(tensor.getFunction())) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
                cudac.forward(tenser.first());
                tenser.forEach(a -> creater.forward(a));
            } else {
                Tensor func = (Tensor) tensor.getFunction();
                cudac.forward(func);
                creater.forward(func);
            }
        } else {
            cudac.forward(tensor);
            creater.forward(tensor);
        }

        String code = cudac.getFuncCode().replace("compute", name);
        System.out.println(code);

        function = createFunction(name, code);
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
        CudaCreater cudac = new CudaCreater();

        if (tensor instanceof TensorFunction) {
            if (BeanUtil.isTenser(tensor.getFunction())) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
                cudac.setBackward(tenser.first());
                tenser.forEach(a -> creater.setBackward(a));
            } else {
                Tensor func = (Tensor) tensor.getFunction();
                cudac.setBackward(func);
                creater.setBackward(func);
            }
        } else {
            cudac.setBackward(tensor);
            creater.setBackward(tensor);
        }

        String code = cudac.getGradCode().replace("gradient", name);
        System.out.println(code);

        function = createFunction(name, code);
        tensor.setCore(creater);
        functions.put(name, function);
        return function;
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