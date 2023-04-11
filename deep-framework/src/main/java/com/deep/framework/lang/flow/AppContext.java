package com.deep.framework.lang.flow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class AppContext implements Operator {

    public AppContext() { }

    public AppContext(Context context, double value, Gradient... inputGrad) {
        this.idx = context.data().size();
        this.data = context.data();
        this.gradMap = context.gradMap();
        this.inputGrad = inputGrad;
        this.value = value;
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

    public List<Consumer> getGradFunc() {
        return gradFunc;
    }

    public void setGradFunc(Consumer func) {
        gradFunc.add(func);
    }

    public double getGrad() {
        return grad;
    }

    public AppContext setGrad(double grad) {
        this.grad = grad;
        return this;
    }

    public void gradient() {
        for (Gradient func : inputGrad) {
            func.apply(grad).gradient();
        }
    }

    private int idx;
    private double value, grad;
    private Gradient[] inputGrad;
    private List<Double> data = new ArrayList<>();
    private List<Consumer> gradFunc = new ArrayList<>();
    private Map<Integer, Double> gradMap = new HashMap();
}