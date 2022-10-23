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
    public String func = "", grad = "", inParams = "", outParams = "", gradParams = "", code = "";
    public String inGradParams = "", outGradParams;
    public Map<String, Integer> inxMap, inxGradMap;

    public TensorCore(){
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
            if (tensor.getName().equals("Tensor::Add")) {
                if (isSame(tensor.getInput())) {
                    forward(tensor.getInput()[0]);
                    computes(tensor);
                }
            } else {
                for (Tensor o : tensor.getInput()) {
                    forward(o);
                }
                compute(tensor);
            }
        }
    }

    public void backward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            forEach(tensor.getFunction(), this::backward);
            for (Tensor o : tensor.getInput()) {
                backward(o);
            }
        } else if (tensor instanceof TensorOperator) {
            if (tensor.getName().equals("Tensor::Add")) {
                gradients(tensor);
                backward(tensor.getInput()[0]);
            } else {
                gradient(tensor);
                for (Tensor o : tensor.getInput()) {
                    backward(o);
                }
            }
        }
    }

    public void compute(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        func = func.concat(functor.compute());
        inParams = inParams.concat(getInputParam(tensor));
        outParams = outParams.concat(output.getValId().trim()).concat(",");

        code = getFuncCode(tensor, outParams);
    }

    public void computes(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();

        code = getFuncCodes(tensor, outParams);
    }

    public void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        grad = grad.concat(functor.gradient(""));
        inGradParams = inGradParams.concat(getGradParam(tensor));
        gradParams = gradParams.concat(output.getGradId().trim()).concat(",");

        code = getGradCode(tensor, outParams, gradParams, inGradParams);
    }

    public void gradients(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        functor.setId(output.getId());
        functor.setInput(tensor.getInput());

        grad = grad.concat(Arrays.stream(tensor.getInput()).filter(a -> !(a instanceof TensorConst)).limit(1).map(a -> (None) a.getOutput()).map(a -> a.getGradId() + "=" + output.getGradId() + ";").collect(Collectors.joining()));
        inGradParams = inGradParams.concat(getGradParam(tensor));
        gradParams = gradParams.concat(output.getGradId().trim()).concat(",");

        code = getGradCodes(tensor, outParams, gradParams, inGradParams);
    }

    private String getFuncCode(Tensor tensor, String outParams) {
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

    private String getFuncCodes(Tensor tensor, String outParams) {
        String[] outParam = getParam(outParams);
        Map<String, String> outMap = new HashMap<>();
        IntStream.range(0, outParam.length).forEach(i -> outMap.put(outParam[i], String.valueOf(i)));

        return getFuncCodes(tensor, outParam, outMap);
    }

    private String getGradCode(Tensor tensor, String outParams, String gradParams, String inGradParams) {
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

    private String getGradCodes(Tensor tensor, String outParams, String gradParams, String inGradParams) {
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

    private String getInputParam(Tensor tensor) {
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

    private String getGradParam(Tensor tensor) {
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

    public String[] getParam(String param) {
        return Arrays.stream(param.split(",")).map(String::trim).distinct().toArray(String[]::new);
    }

    public static boolean isSame(Tensor[] tensor) {
        Tensor m = tensor[0], n = tensor[1];
        TensorCore corem = new TensorCore();
        corem.forward(m);

        TensorCore coren = new TensorCore();
        coren.forward(n);

        return corem.code.equals(coren.code);
    }

    private String getFuncCodes(Tensor tensor, String[] outParam, Map<String, String> outMap) {
        int length = outParam.length, size = tensor.getInput().length;
        String valId = ((None) tensor.getInput()[0].getOutput()).getValId().trim();
        tensor.setDataSize(size * length + 1);
        return new StringBuilder(code)
            .append("extern \"C\" __global__ void compute(double* in, double* out){")
            .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
            .append("int M = ").append(length).append(";")
            .append("compute<<<").append(1).append(",").append(size).append(">>>(in, out);")
            .append("for (int i = 0; i < ").append(size).append("; i++) {")
            .append("out[").append(size * length + 1).append("]+=out[i * M +").append(outMap.get(valId)).append("];")
            .append("}")
        .append("}")
        .toString();
    }

    private String getFuncCode(Tensor tensor, String[] param) {
        return new StringBuilder()
        .append("extern \"C\" __global__ void compute(double* in, double* out){")
            .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
            .append("int M = ").append(param.length).append(";")
            .append(func)
        .append("}")
        .toString();
    }

    private String getGradCode(Tensor tensor, String[] param, String[] outParam) {
        return new StringBuilder()
        .append("extern \"C\" __global__ void gradient(double* in, double* out, double* outGrad, double* inGrad){")
            .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
            .append("int M = ").append(outParam.length).append(";")
            .append(param.length == 0 ? "" : "double " + String.join(",", param) + ";")
            .append(grad)
        .append("}")
        .toString();
    }

    private String getGradCodes(Tensor tensor, String[] param, String[] outParam) {
        return new StringBuilder()
        .append("extern \"C\" __global__ void gradient(double* in, double* out, double* outGrad, double* inGrad){")
            .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
            .append("int M = ").append(outParam.length).append(";")
            .append(param.length == 0 ? "" : "double " + String.join(",", param) + ";")
            .append(grad)
        .append("}")
        .toString();
    }

}