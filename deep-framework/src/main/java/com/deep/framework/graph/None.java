package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

@Data
public class None implements Serializable {

    public None(Tensor tensor) {
        this.tensor = tensor;
        this.gradre = tensor.isGradre();
    }

    public None(Tensor tensor, int idx) {
        this.tensor = tensor;
        this.gradre = tensor.isGradre();
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
        this.gradre = gradre;
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

    public void setGradi(double grad) {
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

    public String getGradId() {
        return "  e" + id + "  ";
    }

    public String getValId() {
        if (gradre) {
            return "  a" + id + "  ";
        }
        return "" + getValue();
    }

    private double value, grad;
    private transient int idx;
    private transient Tensor tensor;
    private transient boolean reduce, gradre;
    private int id = ID.getAndIncrement();
    public transient static AtomicInteger ID = new AtomicInteger();
}