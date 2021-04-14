package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;

@Data
public class None implements Serializable {

    public None(double value) {
        this.value = value;
        this.isGrad = true;
    }

    public None(double value, boolean isGrad) {
        this.value = value;
        this.isGrad = isGrad;
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

    public boolean isGrad() { return isGrad; }

    private double value;
    private Double grad;
    private boolean isGrad;
    private transient boolean reduce;
    
}
