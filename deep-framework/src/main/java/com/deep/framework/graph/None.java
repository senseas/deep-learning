package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

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

    public void setGradx(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad = grad;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setGrad(grad);
        } else {
            ((double[]) tensor.getGrad())[idx] = grad;
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

    public boolean isOut() {
        if (Objects.isNull(tensor)) {
            return this.out;
        } else {
            return tensor.isOut();
        }
    }

    public None setParams(Object... arr) {
        params = new ArrayList<>();
        for (Object o : arr) {
            if (o instanceof List) {
                params.addAll((List) o);
            } else if (o instanceof None) {
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

    public List<None> getParams() {
        if (Objects.nonNull(params) && !params.isEmpty()) return params;
        if (tensor instanceof TensorConst) return Arrays.asList();
        return Arrays.asList(new NoneOut(this));
    }

    public void setGrads(Object... arr) {
        if (Objects.nonNull(params)) return;
        params = new ArrayList<>();
        grads = "";
        None none = getNone(arr);
        for (Object o : arr) {
            if (o instanceof String) {
                grads = grads.concat((String) o);
            } else if (o instanceof None) {
                None a = (None) o;
                if (a == none) {
                    params.addAll(a.getParams());
                    grads = grads.concat(a.getGrads());
                } else {
                    params.add(a);
                    grads = grads.concat("{var}");
                }
            }
        }
    }

    private None getNone(Object[] arr) {
        return (None) Stream.of(arr).filter(a -> a instanceof None).reduce((a, b) -> {
            None m = (None) a;
            if (Objects.nonNull(m.getTensor())) {
                for (Tensor o : m.getTensor().getInput()) {
                    if (o.getOutput().equals(b)) return a;
                }
            }
            return b;
        }).get();
    }

    public void setFuncs(Object... arr) {
        if (Objects.nonNull(paramx)) return;
        paramx = new ArrayList<>();
        paramx.add(this);
        funcs = "({var}=";
        for (Object o : arr) {
            if (o instanceof String) {
                funcs = funcs.concat((String) o);
            } else if (o instanceof None) {
                None a = (None) o;
                paramx.addAll(a.getParamx());
                funcs = funcs.concat(a.getFuncs());
            }
        }
        funcs = funcs.concat(")");
    }

    private int idx;
    private transient Tensor tensor;
    private double value, grad;
    private transient boolean reduce, out;
    private transient String grads = "{var}", funcs = "{var}";
    private transient List<None> paramx, params;
}
