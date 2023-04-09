package com.deep.framework.lang.flow;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class Function implements Operator {

    public Function() {}

    public Function(Context context, Operator... input) {
        this.idx = context.data().size();
        this.data = context.data();
        this.gradMap = context.gradMap();
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

    public List<Consumer> getGradFunc() {
        return gradFunc;
    }

    public void setGradFunc(Consumer func) {
        gradFunc.add(func);
    }

    public double getGrad() {
        return grad;
    }

    public void setGrad(double grad) {
        this.grad = grad;
    }

    public void gradient() {
        for (Operator func : input) {
            func.getGradFunc().forEach(a -> a.accept(0));
            func.gradient();
        }
    }

    private int idx;
    private double value, grad;
    private Operator[] input;
    private List<Double> data = new ArrayList<>();
    private List<Consumer> gradFunc = new ArrayList<>();
    private Map<Integer, Double> gradMap = new HashMap();
}