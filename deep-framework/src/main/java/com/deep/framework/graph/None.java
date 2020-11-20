package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;

@Data
public class None implements Serializable {

    public None(Double value) {
        this.name = "None";
        this.value = value;
    }

    public None(Double value, String name) {
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
    private Double value;
    private Double grad;
    private transient Boolean reduce;
}
