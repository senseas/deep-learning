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
    public Map<String, TensorFunctor> map = new HashMap<>();
    public String func = "", grad = "", code = "";
    public List<String> inParams = new ArrayList<>(), outParams = new ArrayList<>(), gradParams = new ArrayList<>();
    public List<String> inGradParams = new ArrayList<>(), outGradParams = new ArrayList<>();
    public Map<String, Integer> inxMap, inxGradMap;

    public TensorCore() {
        TensorCompiler tc = new TensorCompiler();
        Method[] methods = tc.getClass().getDeclaredMethods();
        Arrays.stream(methods).forEach(method -> {
            try {
                Class type = (Class) method.getGenericParameterTypes()[0];
                Tensor[] args = IntStream.range(0, method.getParameterCount()).mapToObj(a -> new Tensor(new int[]{1})).toArray(Tensor[]::new);
                TensorFunctor tensor = (TensorFunctor) method.invoke(tc, type.isArray() ? new Object[]{args} : args);
                map.put(tensor.getName(), tensor);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    public void forward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            forEach(tensor.getFunction(), this::forward);
        } else if (tensor instanceof TensorOperator) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            compute(tensor);
        }
    }

    public void backward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            forEach(tensor.getFunction(), this::backward);
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

    public void compute(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        func = func.concat(functor.compute());
        inParams.addAll(getInputParam(tensor));
        inParams = inParams.stream().distinct().collect(Collectors.toList());

        outParams.add(output.getValId().trim());
        outParams = outParams.stream().distinct().collect(Collectors.toList());

        code = getFuncCode(tensor);
    }

    private String getFuncCode(Tensor tensor) {
        Map<String, Integer> inMap = new HashMap<>();
        IntStream.range(0, inParams.size()).forEach(i -> inMap.put(inParams.get(i), i));

        Map<String, Integer> outMap = new HashMap<>();
        IntStream.range(0, outParams.size()).forEach(i -> outMap.put(outParams.get(i), i));

        String codes = getFuncCode();
        return getFuncCodex(inMap, outMap, codes);
    }

    private String getFuncCode() {
        return new StringBuilder()
        .append("extern \"C\" __global__ void compute(double* in, double* out){")
            .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
            .append("int M = ").append(outParams.size()).append(";")
            .append(func)
        .append("}").toString();
    }

    private String getFuncCodex(Map<String, Integer> inMap, Map<String, Integer> outMap, String codes) {
        Map<String, Integer> inMapx = Optional.ofNullable(inxMap).orElse(inMap);
        return Arrays.stream(codes.split("  ")).map(a -> {
            if (Objects.nonNull(inMapx.get(a))) return "in[idx+" + inMapx.get(a) + "]";
            if (Objects.nonNull(outMap.get(a))) return "out[idx * M +" + outMap.get(a) + "]";
            return a;
        }).collect(Collectors.joining(""));
    }

    public void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        grad = grad.concat(functor.gradient(""));
        inGradParams.addAll(getGradParam(tensor));
        inGradParams = inGradParams.stream().distinct().collect(Collectors.toList());

        gradParams.add(output.getGradId().trim());
        gradParams = gradParams.stream().distinct().collect(Collectors.toList());

        code = getGradCode(tensor);
    }

    private String getGradCode(Tensor tensor) {
        Map<String, String> outMap = new HashMap<>();
        IntStream.range(0, outParams.size()).forEach(i -> outMap.put(outParams.get(i), String.valueOf(i)));

        Map<String, String> outGradMap = new HashMap<>();
        IntStream.range(0, outGradParams.size()).forEach(i -> outGradMap.put(outGradParams.get(i), String.valueOf(i)));

        String codes = getGradCode(gradParams);
        return getGradCodex(outMap, outGradMap, codes);
    }

    private String getGradCode(List<String> gradParams) {
        String params = gradParams.stream().filter(a -> code.contains(a)).collect(Collectors.joining(","));
        return new StringBuilder()
        .append("extern \"C\" __global__ void gradient(double* in, double* out, double* outGrad, double* inGrad){")
            .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
            .append("int M = ").append(outParams.size()).append(";")
            .append(params.isEmpty() ? "" : "double " + params + ";")
            .append(grad)
        .append("}")
        .toString();
    }

    private String getGradCodex(Map<String, String> outMap, Map<String, String> outGradMap, String codes) {
        return Arrays.stream(codes.split("  ")).map(a -> {
            if (Objects.nonNull(inxMap.get(a))) return "in[idx+" + inxMap.get(a) + "]";
            if (Objects.nonNull(inxGradMap.get(a))) return "inGrad[idx +" + inxGradMap.get(a) + "]+";
            if (Objects.nonNull(outMap.get(a))) return "out[idx * M +" + outMap.get(a) + "]";
            if (Objects.nonNull(outGradMap.get(a))) return "outGrad[idx + " + outGradMap.get(a) + "]";
            return a;
        }).collect(Collectors.joining(""));
    }

    private List<String> getInputParam(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).filter(BeanUtil::isNone).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).map(None::getValId).map(String::trim).toList();
    }

    private List<String> getGradParam(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).filter(BeanUtil::isNone).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).map(None::getGradId).map(String::trim).toList();
    }

}