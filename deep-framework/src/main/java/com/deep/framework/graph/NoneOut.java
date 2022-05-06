package com.deep.framework.graph;

import java.util.List;

public class NoneOut extends None {
    private None one;

    public NoneOut(None one) {
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

}