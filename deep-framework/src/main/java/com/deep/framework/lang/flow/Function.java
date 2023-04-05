package com.deep.framework.lang.flow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Function implements Operator {

    public Function() { }

    public Function(Context context, Operator... input) {
        this.data = context.data();
        this.gradMap = context.gradMap();
        this.idx = context.data().size();
        this.input = input;
    }

    public List<Double> data() {
        return data;
    }

    public Map<Integer, Double> gradMap() {
        return gradMap;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getGradx() {
        return gradx;
    }

    public void setGradx(double gradx) {
        this.gradx = gradx;
    }

    public double getGrad() {
        return grad;
    }

    public void setGrad(double grad) {
        this.grad = grad;
    }

    public void gradient() {
        for (Operator func : input) {
            func.setGrad(grad * func.getGradx());
            func.gradient();
        }
    }

    private int idx = 0;
    private double value, gradx, grad;
    private Operator[] input;
    private List<Double> data = new ArrayList<>();
    private Map<Integer, Double> gradMap = new HashMap();
}