package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;

@Data
public class None implements Serializable {

    public None(double value) {
        this.value = value;
    }

    public void setGrad(Double grad) {
        if (this.grad != null && grad != null) {
            this.grad = this.grad + grad;
        } else {
            this.grad = grad;
        }
    }

    public Double getGrad() {
        if (this.grad != null) {
            return grad;
        } else {
            return 1d;
        }
    }

    private double value;
    private Double grad;
    private transient boolean reduce;
}
