package com.deep.framework.bean;

import com.deep.framework.graph.Graph;
import com.deep.framework.graph.Tenser;
import lombok.Data;

@Data
public class None {

    public None(Double value) {
        this.name = "None";
        this.value = value;
        this.graph = new Graph();
    }

    public None(Double value, String name) {
        this.name = name;
        this.value = value;
        this.graph = new Graph();
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
    private transient Graph<Tenser<None>> graph;

}
