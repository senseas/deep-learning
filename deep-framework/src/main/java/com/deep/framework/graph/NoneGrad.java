package com.deep.framework.graph;

import java.util.List;

public class NoneGrad extends None {
    private None one;

    public NoneGrad(None one) {
        super(null);
        this.one = one;
    }

    public double getValue() {
        return one.getGrad();
    }

    public List<None> getParamx() {
        return one.getParamx();
    }

    public List<None> getParams() {
        return one.getParams();
    }

    public String getGrads() {
        return one.getGrads();
    }
}