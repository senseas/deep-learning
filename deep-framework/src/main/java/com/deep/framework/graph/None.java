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
        if (this.gradef) {
            this.grad = this.grad + grad;
        } else {
            this.grad = grad;
            this.gradef = true;
        }
    }

    public void reset() {
        this.reduce = false;
        this.gradef = false;
        this.grad = 1d;
    }

    private double value, grad = 1d;
    private transient boolean gradre, gradef, reduce;

}
