package com.deep.framework.framework;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.forEach;

public abstract class TensorGene implements Serializable {
    public String index = null;

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
            setForwardParams();
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
            setBackwardParams();
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

    public void setForward(Tensor tensor) {
        isForward = true;
        clear();
        forward(tensor);
    }

    public void setBackward(Tensor tensor) {
        isForward = false;
        setOutGrad(tensor);
        clear();
        backward(tensor);
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

    public abstract void compute(Tensor tensor);

    public abstract void gradient(Tensor tensor);

    public void setInParamsx() {
        int index = inParamsx.size() - 1;
        inParamsx.set(index, inParamsx.get(index) + 1);
    }

    public void setOutParamsx() {
        int index = inParamsx.size() - 1;
        outParamsx.set(index, outParamsx.get(index) + 1);
    }

    public void setInBackParamsx() {
        int index = inBackParamsx.size() - 1;
        inBackParamsx.set(index, inBackParamsx.get(index) + 1);
    }

    public void setOutBackParamsx() {
        int index = outBackParamsx.size() - 1;
        outBackParamsx.set(index, outBackParamsx.get(index) + 1);
    }

    public void setInGradParamsx() {
        int index = inGradParamsx.size() - 1;
        inGradParamsx.set(index, inGradParamsx.get(index) + 1);
    }

    public void setOutGradParamsx() {
        int index = outGradParamsx.size() - 1;
        outGradParamsx.set(index, outGradParamsx.get(index) + 1);
    }

    public void setForParamsx(int x) {
        int index = inParamsx.size() - 1;
        inParamsx.set(index, x * inParamsx.get(index));

        int outdex = outParamsx.size() - 1;
        outParamsx.set(outdex, x * outParamsx.get(outdex));
    }

    public void setBackParamsx(int x) {
        int index = inBackParamsx.size() - 1;
        inBackParamsx.set(index, x * inBackParamsx.get(index));

        int outdex = outBackParamsx.size() - 1;
        outBackParamsx.set(outdex, x * outBackParamsx.get(outdex));

        int inGraddex = inGradParamsx.size() - 1;
        inGradParamsx.set(inGraddex, x * inGradParamsx.get(inGraddex));

        int outGraddex = outGradParamsx.size() - 1;
        outGradParamsx.set(outGraddex, x * outGradParamsx.get(outGraddex));
    }

    public void setForwardParams() {
        inParamsx.add(0);
        outParamsx.add(0);
    }

    public void setBackwardParams() {
        inBackParamsx.add(0);
        outBackParamsx.add(0);
        inGradParamsx.add(0);
        outGradParamsx.add(0);
    }

    public Map<String, TensorFunctor> map = new HashMap<>();

    public List<None> inParams = new ArrayList<>(), outParams = new ArrayList<>();
    public List<None> inBackParams = new ArrayList<>(), outBackParams = new ArrayList<>();
    public List<None> inGradParams = new ArrayList<>(), outGradParams = new ArrayList<>();
    public Set<String> innerGradParam = new HashSet<>();

    public List<Integer> inParamsx = new ArrayList<>(), outParamsx = new ArrayList<>();
    public List<Integer> inBackParamsx = new ArrayList<>(), outBackParamsx = new ArrayList<>();
    public List<Integer> inGradParamsx = new ArrayList<>(), outGradParamsx = new ArrayList<>();

    public AtomicInteger idxIn = new AtomicInteger(), idxOut = new AtomicInteger();
    public AtomicInteger idxInGrad = new AtomicInteger(), idxOutGrad = new AtomicInteger();
    public boolean isForward;
}