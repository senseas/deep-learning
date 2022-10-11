package com.deep.framework.framework;

import com.deep.framework.graph.*;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.forEach;

public class TensorCore implements Serializable {
    public static Map<String, TensorFunctor> map = new HashMap<>();
    public static String func = "", grad = "";
    public static String fparam = "", gparam = "";
    public static String code = "";

    static {
        TensorCompiler tc = new TensorCompiler();
        Method[] methods = tc.getClass().getDeclaredMethods();
        Arrays.stream(methods).forEach(method -> {
            try {
                Class type = (Class) method.getGenericParameterTypes()[0];
                Tensor[] args = IntStream.range(0, method.getParameterCount()).mapToObj(a -> new Tensor(new int[]{1})).toArray(Tensor[]::new);
                TensorFunctor tensor = (TensorFunctor) method.invoke(tc, type.isArray() ? new Object[]{args} : args);
                TensorCore.map.put(tensor.getName(), tensor);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    public synchronized static void forward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            forEach(tensor.getFunction(), TensorCore::forward);
        } else if (tensor instanceof TensorOperator) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            compute(tensor);
        }
    }

    public synchronized static void backward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            forEach(tensor.getFunction(), TensorCore::backward);
            for (Tensor o : tensor.getInput()) {
                backward(o);
            }
        } else if (tensor instanceof TensorOperator) {
            gradient(tensor);
            for (Tensor o : tensor.getInput()) {
                backward(o);
            }
        }
    }

    public synchronized static void compute(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        func = func.concat(functor.compute());
        if (fparam.isEmpty()) fparam = fparam.concat(getInputParam(tensor)).concat(",");
        fparam = fparam.concat(output.getValId()).concat(",");

        code = getFuncCode(tensor, fparam);
    }

    public synchronized static void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());
        grad = grad.concat(functor.gradient(""));
        gparam = gparam.concat(output.getGradId()).concat(",");
        code = getGradCode(tensor, fparam, gparam);
    }

    private static String getFuncCode(Tensor tensor, String fparam) {
        String[] param = fparam.split(",");
        Map<String, String> map = new HashMap<>();
        IntStream.range(0, param.length).forEach(i -> map.put(param[i].trim(), String.valueOf(i)));

        String code = getFuncCode(tensor, param);
        return Arrays.stream(code.split(" ")).map(a -> Objects.nonNull(map.get(a)) ? "data[idx * M +".concat(map.get(a)).concat("]") : a).collect(Collectors.joining(" "));
    }

    private static String getGradCode(Tensor tensor, String fparam, String gparam) {
        String[] param = Arrays.stream(gparam.split(",")).map(String::trim).toArray(String[]::new);
        String[] dataParam = fparam.concat(param[0]).split(",");
        Map<String, String> dataMap = new HashMap<>();
        IntStream.range(0, dataParam.length).forEach(i -> dataMap.put(dataParam[i].trim(), String.valueOf(i)));

        String[] gradParam = getGradParam(tensor);
        Map<String, String> gradMap = new HashMap<>();
        IntStream.range(0, gradParam.length).forEach(i -> gradMap.put(gradParam[i].trim(), String.valueOf(i)));

        String code = getGradCode(tensor, param, dataParam);
        return Arrays.stream(code.split(" ")).map(a -> Objects.nonNull(gradMap.get(a)) ? "grad[idx]+" : a).map(a -> Objects.nonNull(dataMap.get(a)) ? "data[idx * M +".concat(dataMap.get(a)).concat("]") : a).collect(Collectors.joining(" "));
    }

    private static String getFuncCode(Tensor tensor, String[] param) {
        return "extern \"C\" __global__ void " + tensor.getName() + "(double* data)" + "{" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + param.length + ";" +
            func +
        "}";
    }

    private static String getGradCode(Tensor tensor, String[] param, String[] dataParam) {
        return "extern \"C\" __global__ void " + tensor.getName() + "(double* data, double* grad)" + "{" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + dataParam.length + ";" +
            "double " + String.join(",", Arrays.copyOfRange(param, 1, param.length)) + ";" +
            grad +
        "}";
    }

    private static String getInputParam(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).map(a -> ((None) a.getOutput())).filter(None::isGradre).map(None::getValId).collect(Collectors.joining(","));
    }

    private static String[] getGradParam(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).map(a -> ((None) a.getOutput())).filter(None::isGradre).map(None::getGradId).toArray(String[]::new);
    }

}