package com.deep.framework.framework;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.forEach;

public abstract class TensorGene implements Serializable {

    public TensorGene() {
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

    protected void setInputParam(Tensor tensor) {
        Arrays.stream(tensor.getInput()).forEach(a -> {
            if (BeanUtil.isTenser(a.getOutput())) {
                Tenser<None> output = a.getOutput();
                output.stream().forEach(out -> out.setCore(this));
            } else {
                None out = a.getOutput();
                out.setCore(this);
            }
        });
    }

    public void setBackward(Tensor tensor) {
        setOutGrad(tensor);
        backward(tensor);
    }

    private void setOutGrad(Tensor tensor) {
        if (tensor instanceof TensorOperator) {
            forEach(tensor.getOutput(), (None out) -> out.setOutGrad(true));
        } else if (tensor instanceof TensorFunction) {
            forEach(tensor.getFunction(), this::setOutGrad);
        }
    }

    public String getIndex() {
        if (index == null) return "";
        return "+" + index;
    }

    public abstract void compute(Tensor tensor);

    public abstract void gradient(Tensor tensor);

    public Map<String, TensorFunctor> map = new HashMap<>();

    public List<None> inParams = new ArrayList<>(), outParams = new ArrayList<>();
    public List<None> inGradParams = new ArrayList<>(), outGradParams = new ArrayList<>();
    public Set<String> innerGradParam = new HashSet<>();
    public String index = null;
}