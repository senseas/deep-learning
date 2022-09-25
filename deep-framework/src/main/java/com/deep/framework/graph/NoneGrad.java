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

    public int getId() {
        return one.getId();
    }

    public List<None> getFuncx() {
        return one.getFuncx();
    }

    public List<None> getGradx() {
        return one.getGradx();
    }

    public String getGradc() {
        return one.getGradc();
    }

    public String getFunc() {
        return one.getFunc();
    }

    public String getParam() {
        return one.getParam();
    }

}