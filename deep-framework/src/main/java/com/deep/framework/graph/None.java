package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

@Data
public class None implements Serializable {

    public None(double[] value, double[] grad, boolean[] reduce) {
        this.value = value;
        this.grad = grad;
        this.reduce = reduce;
    }

    public None(double[] value, double[] grad, boolean[] reduce, int idx) {
        this.value = value;
        this.grad = grad;
        this.reduce = reduce;
        this.idx = idx;
    }

    public None(double value) {
        this.value = new double[]{value};
        this.grad = new double[]{0d};
        this.reduce = new boolean[]{false};
    }

    public None(double value, boolean gradre) {
        this.value = new double[]{value};
        this.grad = new double[]{0d};
        this.reduce = new boolean[]{false};
    }

    public double getValue() {
        return this.value[idx];
    }

    public void setValue(double value) {
        this.value[idx] = value;
    }

    public double getGrad() {
        return this.grad[idx];
    }

    public void setGrad(double grad) {
        this.grad[idx] += grad;
    }

    public boolean isReduce() {
        return this.reduce[idx];
    }

    public void setReduce(boolean reduce) {
        this.reduce[idx] = reduce;
    }

    public void reset() {
        this.reduce[idx] = false;
        this.grad[idx] = 0d;
    }

    public String getFuncs() {
        if (tensor instanceof TensorConst)
            return String.valueOf(getValue());
        return funcs;
    }

    public List<None> getParamx() {
        if (tensor instanceof TensorConst) return Arrays.asList();
        if (Objects.nonNull(paramx) && !paramx.isEmpty()) return paramx;
        return Arrays.asList(this);
    }

    public List<None> getParams() {
        if (tensor instanceof TensorConst) return Arrays.asList();
        if (Objects.nonNull(params) && !params.isEmpty()) return params;
        return Arrays.asList(this);
    }

    public void setFuncs(Object... arr) {
        if (Objects.nonNull(paramx)) return;
        paramx = new ArrayList<>();
        paramx.add(this);
        funcs = "";
        for (Object o : arr) {
            if (o instanceof String) {
                funcs = funcs.concat((String) o);
            } else if (o instanceof None) {
                None a = (None) o;
                if (a.getTensor() instanceof TensorConst) {
                    funcs = funcs.concat(a.getFuncs());
                } else {
                    paramx.addAll(a.getParamx());
                    funcs = funcs.concat(a.getFuncs());
                }
            }
        }
        funcs = "({var}=".concat(funcs).concat(")");
    }

    public void setGrads(Object... arr) {
        if (Objects.nonNull(params)) return;
        params = new ArrayList<>();
        grads = "";
        for (Object o : arr) {
            if (o instanceof String) {
                grads = grads.concat((String) o);
            } else if (o instanceof None) {
                None a = (None) o;
                if (a.getTensor() instanceof TensorConst) {
                    grads = grads.concat(String.valueOf(a.getValue()));
                } else if (a instanceof NoneGrad) {
                    params.addAll(a.getParams());
                    grads = grads.concat(a.getGrads());
                } else {
                    params.add(a);
                    grads = grads.concat("{var}");
                }
            }
        }
    }

    public None grad() {
        NoneGrad noneGrad = new NoneGrad(this);
        if (Objects.isNull(params)) {
            params = Arrays.asList(noneGrad);
        }
        return noneGrad;
    }

    private int idx;
    private transient Tensor tensor;
    private double[] value, grad;
    private transient boolean[] reduce;
    private transient String grads = "{var}", funcs = "{var}";
    private transient List<None> paramx, params;
}
