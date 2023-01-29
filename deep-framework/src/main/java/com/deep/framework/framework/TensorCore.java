package com.deep.framework.framework;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.deep.framework.lang.ForEach.forEach;

public class TensorCore implements Serializable {
    public Map<String, TensorFunctor> map = new HashMap<>();
    public String func = "", grad = "", code = "";
    public List<None> inParams = new ArrayList<>(), outParams = new ArrayList<>(), gradParams = new ArrayList<>();
    public List<None> inGradParams = new ArrayList<>(), outGradParams = new ArrayList<>();
    public Map<String, Integer> inxMap, inxGradMap;
    public AtomicInteger idxIn = new AtomicInteger(), idxOut = new AtomicInteger();
    public AtomicInteger idxInGrad = new AtomicInteger(), idxOutGrad = new AtomicInteger();
    public boolean isBackward;

    public TensorCore(Integer... inputSize) {
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
        output.setValId(null);
        output.setCore(this);
        getInputParam(tensor);
        /*inParams.addAll(getInputParam(tensor));
        outParams.add(output);*/

        functor.setInput(tensor.getInput());
        functor.setOut(output);

        func = func.concat(functor.compute());
        code = getFuncCode();
    }

    private String getFuncCode() {
        return new StringBuilder()
        .append("extern \"C\" __global__ void compute(double* in, double* out){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("int M = ").append(outParams.size()).append(";")
        .append("int N = ").append(inParams.size()).append(";")
        .append(func)
        .append("}").toString();
    }

    public void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        output.setCore(this);
        getGradParam(tensor);
        /*inGradParams.addAll(getGradParam(tensor));
        gradParams.add(output);*/

        functor.setInput(tensor.getInput());
        functor.setOut(output);

        grad = grad.concat(functor.gradient(""));
        code = getGradCode();
    }

    private String getGradCode() {
        return new StringBuilder()
        .append("extern \"C\" __global__ void gradient(double* in, double* out, double* outGrad, double* inGrad){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("int M = ").append(outParams.size()).append(";")
        .append("int N = ").append(inParams.size()).append(";")
        .append("int X = ").append(inGradParams.size()).append(";")
        .append("int Y = ").append(outGradParams.size()).append(";")
        .append(grad)
        .append("}").toString();
    }

    private List<None> getInputParam(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).map(a -> {
            a.setCore(this);
            return a;
        }).toList();
    }

    private List<None> getGradParam(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).flatMap(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                return output.stream();
            } else {
                None out = a.getOutput();
                return Stream.of(out);
            }
        }).map(a -> {
            a.setCore(this);
            return a;
        }).toList();
    }

    public void setBackward(Tensor tensor) {
        isBackward = true;
        setOutGrad(tensor);
        clear();
    }

    public void setForward() {
        isBackward = false;
        clear();
    }

    private void clear() {
        idxIn = new AtomicInteger();
        idxOut = new AtomicInteger();
        idxInGrad = new AtomicInteger();
        idxOutGrad = new AtomicInteger();
    }

    private void setOutGrad(Tensor tensor) {
        if (tensor instanceof TensorOperator) {
            forEach(tensor.getOutput(), (None out) -> out.setOutGrad(true));
        } else if (tensor instanceof TensorFunction) {
            forEach(tensor.getFunction(), this::setOutGrad);
        }
    }
}