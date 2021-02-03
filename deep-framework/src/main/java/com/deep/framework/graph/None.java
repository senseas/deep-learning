package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;

@Data
public class None implements Serializable {

    public None(double value) {
        this.name = "None";
        this.value = value;
    }

    public None(double value, String name) {
        this.name = name;
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

    private String name;
    private double value;
    private Double grad;
    private transient boolean reduce;
}
