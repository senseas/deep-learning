package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

@Data
public class None implements Serializable {

    public None(Tensor tensor) {
        this.tensor = tensor;
    }

    public None(Tensor tensor, int idx) {
        this.tensor = tensor;
        this.idx = idx;
    }

    public None(double value) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
    }

    public None(double value, boolean gradre) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
    }

    public double getValue() {
        if (Objects.isNull(tensor)) {
            return this.value;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getValue();
        } else {
            return ((double[]) tensor.getValue())[idx];
        }
    }

    public void setValue(double value) {
        if (Objects.isNull(tensor)) {
            this.value = value;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setValue(value);
        } else {
            ((double[]) tensor.getValue())[idx] = value;
        }
    }

    public double getGrad() {
        if (Objects.isNull(tensor)) {
            return this.grad;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getGrad();
        } else {
            return ((double[]) tensor.getGrad())[idx];
        }
    }

    public void setGrad(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad += grad;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setGrad((double) tensor.getGrad() + grad);
        } else {
            ((double[]) tensor.getGrad())[idx] += grad;
        }
    }

    public boolean isReduce() {
        if (Objects.isNull(tensor)) {
            return this.reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            return (boolean) tensor.getReduce();
        } else {
            return ((boolean[]) tensor.getReduce())[idx];
        }
    }

    public void setReduce(boolean reduce) {
        if (Objects.isNull(tensor)) {
            this.reduce = reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReduce(reduce);
        } else {
            ((boolean[]) tensor.getReduce())[idx] = reduce;
        }
    }

    public void resetx() {
        this.grad = 0d;
    }

    public void reset() {
        if (Objects.isNull(tensor)) {
            this.reduce = false;
            this.grad = 0d;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReduce(false);
            tensor.setGrad(0d);
        } else {
            ((boolean[]) tensor.getReduce())[idx] = false;
            ((double[]) tensor.getGrad())[idx] = 0d;
        }
    }

    public None setParams(Object... arr) {
        params = new ArrayList<>();
        for (Object o : arr) {
            if (o instanceof List) {
                params.addAll((List) o);
            } else {
                params.add((None) o);
            }
        }
        return this;
    }

    public None setParamx(Object... arr) {
        paramx = new ArrayList<>();
        for (Object o : arr) {
            if (o instanceof List) {
                paramx.addAll((List) o);
            } else if (!(o instanceof TensorConst)) {
                paramx.add((None) o);
            }
        }
        return this;
    }

    public String getFuncs() {
        if (tensor instanceof TensorConst)
            return String.valueOf(getValue());
        return funcs;
    }

    public List<None> getParamx() {
        if (Objects.nonNull(paramx) && !paramx.isEmpty()) return paramx;
        if (tensor instanceof TensorConst) return Arrays.asList();
        return Arrays.asList(this);
    }

    private int idx;
    private transient Tensor tensor;
    private double value, grad;
    private transient boolean reduce;
    private String grads = "{var}";
    private String funcs = "{var}";
    private List<None> params = new ArrayList<>();
    private List<None> paramx;
}
