package com.deep.framework.framework;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.deep.framework.lang.ForEach.forEach;

public class TensorCore implements Serializable {
    public static Map<String, TensorFunctor> map = new HashMap<>();
    public static String func = "", grad = "", fparam = "", gparam = "", code = "";
    public static String fparamx = "", gparamx = "", gradout, next;

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

    public static void forward(Tensor tensor) {
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

    public static void backward(Tensor tensor) {
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

    public static void compute(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        func = func.concat(functor.compute());
        fparam = fparam.concat(getInputParam(tensor)).concat(output.getValId().trim()).concat(",");

        code = getFuncCode(tensor, fparam);
    }

    public static void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        grad = grad.concat(functor.gradient(""));
        gparam = gparam.concat(output.getGradId().trim()).concat(",");
        gparamx = gparamx.concat(getGradParam(tensor));

        code = getGradCode(tensor, fparam, gparam, gparamx);
    }

    private static String getFuncCode(Tensor tensor, String fparam) {
        String[] param = getParam(fparam);
        Map<String, String> map = new HashMap<>();
        IntStream.range(0, param.length).forEach(i -> map.put(param[i].trim(), String.valueOf(i)));

        String code = getFuncCode(tensor, param);
        return Arrays.stream(code.split("  ")).map(a -> Objects.nonNull(map.get(a)) ? "data[idx * M +".concat(map.get(a)).concat("]") : a).collect(Collectors.joining(""));
    }

    private static String getGradCode(Tensor tensor, String fparam, String gparam, String gparamx) {
        String[] dataParam = getParam(fparam);
        Map<String, String> dataMap = new HashMap<>();
        IntStream.range(0, dataParam.length).forEach(i -> dataMap.put(dataParam[i].trim(), String.valueOf(i)));

        String[] gradParam = getParam(gparamx);
        Map<String, String> gradMap = new HashMap<>();
        IntStream.range(0, gradParam.length).forEach(i -> gradMap.put(gradParam[i].trim(), String.valueOf(i)));

        String[] gradNextParam = getParam(next);
        Map<String, String> gradNextMap = new HashMap<>();
        IntStream.range(0, gradNextParam.length).forEach(i -> gradNextMap.put(String.valueOf(i),gradNextParam[i].trim()));

        String[] gradOutParam = getParam(gradout);
        Map<String, String> gradOutMap = new HashMap<>();
        IntStream.range(0, gradOutParam.length).forEach(i -> gradOutMap.put(gradOutParam[i].trim(), String.valueOf(i)));

        String code = getGradCode(tensor, getParam(gparam), dataParam, gradParam);
        System.out.println(code);
        return Arrays.stream(code.split("  ")).map(a -> Objects.nonNull(gradMap.get(a)) ? "grad[idx+" + gradNextMap.get(gradMap.get(a)) + "]" : a).map(a -> Objects.nonNull(gradOutMap.get(a)) ? "gradx[idx]" : a).map(a -> Objects.nonNull(dataMap.get(a)) ? "data[idx * M +".concat(dataMap.get(a)).concat("]") : a).collect(Collectors.joining(""));
    }

    private static String getInputParam(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).map(None::getValId).collect(Collectors.joining(",")).concat(",");
    }

    private static String getGradParam(Tensor tensor) {
        List<String> list = Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).filter(BeanUtil::isNone).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).map(None::getGradId).toList();
        return list.isEmpty() ? "" : String.join(",", list).concat(",");
    }

    public static String[] getParam(String param) {
        return Arrays.stream(param.split(",")).map(String::trim).distinct().toArray(String[]::new);
    }

    private static String getFuncCode(Tensor tensor, String[] param) {
        return "extern \"C\" __global__ void compute(double* data)" + "{" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + param.length + ";" +
            func +
        "}";
    }

    private static String getGradCode(Tensor tensor, String[] param, String[] dataParam, String[] gradParam) {
        return "extern \"C\" __global__ void gradient(double* data, double* gradx, double* grad)" + "{" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + dataParam.length + ";" +
            "double " + String.join(",", param) + ";" +
            grad +
        "}";
    }

}