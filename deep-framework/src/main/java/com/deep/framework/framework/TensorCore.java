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
    public static String func = "", grad = "", inParams = "", outParams = "", gradParams = "", code = "";
    public static String inGradParams = "", outGradParams;
    public static Map<String, Integer> inxMap, inxGradMap;

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
        inParams = inParams.concat(getInputParam(tensor));
        outParams = outParams.concat(output.getValId().trim()).concat(",");

        code = getFuncCode(tensor, outParams);
    }

    public static void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        grad = grad.concat(functor.gradient(""));
        inGradParams = inGradParams.concat(getGradParam(tensor));
        gradParams = gradParams.concat(output.getGradId().trim()).concat(",");

        code = getGradCode(tensor, outParams, gradParams, inGradParams);
    }

    private static String getFuncCode(Tensor tensor, String outParams) {
        String[] inParam = getParam(inParams);
        Map<String, String> inMap = new HashMap<>();
        IntStream.range(0, inParam.length).forEach(i -> inMap.put(inParam[i], String.valueOf(i)));

        String[] outParam = getParam(outParams);
        Map<String, String> outMap = new HashMap<>();
        IntStream.range(0, outParam.length).forEach(i -> outMap.put(outParam[i], String.valueOf(i)));

        String codes = getFuncCode(tensor, outParam);
        return Arrays.stream(codes.split("  ")).map(a -> {
            if (Objects.nonNull(inxMap) && Objects.nonNull(inxMap.get(a))) return "in[idx+" + inxMap.get(a) + "]";
            if (Objects.isNull(inxMap) && Objects.nonNull(inMap.get(a))) return "in[idx+" + inMap.get(a) + "]";
            if (Objects.nonNull(outMap.get(a))) return "out[idx * M +" + outMap.get(a) + "]";
            return a;
        }).collect(Collectors.joining(""));
    }

    private static String getGradCode(Tensor tensor, String outParams, String gradParams, String inGradParams) {
        String[] outParam = getParam(outParams);
        Map<String, String> outMap = new HashMap<>();
        IntStream.range(0, outParam.length).forEach(i -> outMap.put(outParam[i], String.valueOf(i)));

        String[] outGradParam = getParam(outGradParams);
        Map<String, String> outGradMap = new HashMap<>();
        IntStream.range(0, outGradParam.length).forEach(i -> outGradMap.put(outGradParam[i], String.valueOf(i)));

        String[] param = Arrays.stream(getParam(gradParams)).filter(a -> code.contains(a)).toArray(String[]::new);
        String codes = getGradCode(tensor, param, outParam);
        return Arrays.stream(codes.split("  ")).map(a -> {
            if (Objects.nonNull(inxMap.get(a))) return "in[idx+" + inxMap.get(a) + "]";
            if (Objects.nonNull(inxGradMap.get(a))) return "inGrad[idx +" + inxGradMap.get(a) + "]+";
            if (Objects.nonNull(outMap.get(a))) return "out[idx * M +" + outMap.get(a) + "]";
            if (Objects.nonNull(outGradMap.get(a))) return "outGrad[idx + " + outGradMap.get(a) + "]";
            return a;
        }).collect(Collectors.joining(""));
    }

    private static String getInputParam(Tensor tensor) {
        List<String> list = Arrays.stream(tensor.getInput()).filter(BeanUtil::isNone).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).map(None::getValId).toList();
        return list.isEmpty() ? "" : String.join(",", list).concat(",");
    }

    private static String getGradParam(Tensor tensor) {
        List<String> list = Arrays.stream(tensor.getInput()).filter(BeanUtil::isNone).flatMap(a -> {
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

    public static void forwardClear() {
        func = code = inParams = outParams = "";
    }

    public static void backwardClear() {
        grad = code = gradParams = "";
    }

    private static String getFuncCode(Tensor tensor, String[] param) {
        return "extern \"C\" __global__ void compute(double* in, double* out){" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + param.length + ";" +
            func +
        "}";
    }

    private static String getGradCode(Tensor tensor, String[] param, String[] outParam) {
        String params = String.join(",", param);
        params = params.isEmpty() ? "" : "double " + params + ";";
        return "extern \"C\" __global__ void gradient(double* in, double* out, double* outGrad, double* inGrad){" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + outParam.length + ";" +
            params +
            grad +
        "}";
    }

}