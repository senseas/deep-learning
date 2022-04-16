/*
package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;
import java.util.Objects;

@Data
public class Tensor implements Serializable {

    public Tensor(Tensor tensor) {
        this.tensor = tensor;
    }

    public Tensor(Tensor tensor, int idx) {
        this.tensor = tensor;
        this.idx = idx;
    }

    public Tensor(double value) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
    }

    public Tensor(double value, boolean gradre) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
    }

    public double getValue() {
        if (Objects.isNull(tensor)) {
            return this.value;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getValuex();
        } else {
            return ((double[]) tensor.getValuex())[idx];
        }
    }

    public void setValue(double value) {
        if (Objects.isNull(tensor)) {
            this.value = value;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setValuex(value);
        } else {
            ((double[]) tensor.getValuex())[idx] = value;
        }
    }

    public double getGrad() {
        if (Objects.isNull(tensor)) {
            return this.grad;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getGradx();
        } else {
            return ((double[]) tensor.getGradx())[idx];
        }
    }

    public void setGrad(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad += grad;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setGradx((double) tensor.getGradx() + grad);
        } else {
            ((double[]) tensor.getGradx())[idx] += grad;
        }
    }

    public boolean isReduce() {
        if (Objects.isNull(tensor)) {
            return this.reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            return (boolean) tensor.getReducex();
        } else {
            return ((boolean[]) tensor.getReducex())[idx];
        }
    }

    public void setReduce(boolean reduce) {
        if (Objects.isNull(tensor)) {
            this.reduce = reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReducex(reduce);
        } else {
            ((boolean[]) tensor.getReducex())[idx] = reduce;
        }
    }

    public void reset() {
        if (Objects.isNull(tensor)) {
            this.reduce = false;
            this.grad = 0d;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReducex(false);
            tensor.setGradx(0d);
        } else {
            ((boolean[]) tensor.getReducex())[idx] = false;
            ((double[]) tensor.getGradx())[idx] = 0d;
        }
    }

    private int idx;
    private transient Tensor tensor;
    private double value, grad;
    private transient boolean reduce;
}
*/
