package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;

@Data
public class None implements Serializable {

    public None(double value) {
        this.value = value;
        this.gradre = true;
    }

    public None(double value, boolean isGrad) {
        this.value = value;
        this.gradre = isGrad;
    }

    public void setGrad(double grad) {
        this.grad += grad;
    }

    public void reset() {
        this.reduce = false;
        this.grad = 0d;
    }

    private double value, grad = 0d;
    private transient boolean gradre, reduce;

}
