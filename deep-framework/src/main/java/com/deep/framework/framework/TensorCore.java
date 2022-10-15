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
    public static String fparamx = "", gparamx = "", gradout;
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
        fparam = fparam.concat(output.getValId().trim()).concat(",");
        fparamx = fparamx.concat(getInputParam(tensor));

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
        String[] inParam = getParam(fparamx);
        Map<String, String> inMap = new HashMap<>();
        IntStream.range(0, inParam.length).forEach(i -> inMap.put(inParam[i].trim(), String.valueOf(i)));

        String[] outParam = getParam(fparam);
        Map<String, String> outMap = new HashMap<>();
        IntStream.range(0, outParam.length).forEach(i -> outMap.put(outParam[i].trim(), String.valueOf(i)));

        String codes = getFuncCode(tensor, outParam);
        String code = Arrays.stream(codes.split("  "))
        .map(a -> Objects.nonNull(inMap.get(a)) ? "in[idx+" + inxMap.get(a) + "]" : a)
        .map(a -> Objects.nonNull(outMap.get(a)) ? "out[idx * M +" + outMap.get(a) + "]" : a)
        .collect(Collectors.joining(""));

        return code;
    }

    private static String getGradCode(Tensor tensor, String fparam, String gparam, String gparamx) {
        String[] inParam = getParam(fparamx);
        Map<String, String> inMap = new HashMap<>();
        IntStream.range(0, inParam.length).forEach(i -> inMap.put(inParam[i].trim(), String.valueOf(i)));

        String[] outParam = getParam(fparam);
        Map<String, String> outMap = new HashMap<>();
        IntStream.range(0, outParam.length).forEach(i -> outMap.put(outParam[i].trim(), String.valueOf(i)));

        String[] outGradParam = getParam(gradout);
        Map<String, String> outGradMap = new HashMap<>();
        IntStream.range(0, outGradParam.length).forEach(i -> outGradMap.put(outGradParam[i].trim(), String.valueOf(i)));

        String[] inGradParam = getParam(gparamx);
        Map<String, String> inGradMap = new HashMap<>();
        IntStream.range(0, inGradParam.length).forEach(i -> inGradMap.put(inGradParam[i].trim(), String.valueOf(i)));

        String codes = getGradCode(tensor, getParam(gparam), outParam, inGradParam);
        String code = Arrays.stream(codes.split("  "))
        .map(a -> Objects.nonNull(inMap.get(a)) ? "in[idx+" + inxMap.get(a) + "]" : a)
        .map(a -> Objects.nonNull(outMap.get(a)) ? "out[idx * M +" + outMap.get(a) + "]" : a)
        .map(a -> Objects.nonNull(outGradMap.get(a)) ? "outGrad[idx + " + outGradMap.get(a) + "]" : a)
        .map(a -> Objects.nonNull(inGradMap.get(a)) ? "inGrad[idx +" + inGradMap.get(a) + "]+" : a)
        .collect(Collectors.joining(""));

        return code;
    }

    private static String getInputParam(Tensor tensor) {
        List<String> list = Arrays.stream(tensor.getInput()).filter(Tensor::isGradre).filter(BeanUtil::isNone).flatMap(a -> {
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
        return "extern \"C\" __global__ void compute(double* in, double* out){" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + param.length + ";" +
            func +
        "}";
    }

    private static String getGradCode(Tensor tensor, String[] param, String[] dataParam, String[] gradParam) {
        return "extern \"C\" __global__ void gradient(double* in, double* out, double* outGrad, double* inGrad){" +
            "int idx = blockDim.x * blockIdx.x + threadIdx.x;" +
            "int M = " + dataParam.length + ";" +
            "double " + String.join(",", param) + ";" +
            grad +
        "}";
    }

}