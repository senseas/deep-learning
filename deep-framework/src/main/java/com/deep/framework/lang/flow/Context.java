package com.deep.framework.lang.flow;

import lombok.Data;

import java.util.Objects;

@Data
public class Context {

    public Context(Gradient gradient, double value, Context... input) {
        this.gradient = gradient;
        this.value = value;
        this.input = input;
    }

    public Context(double value) {
        this.value = value;
    }

    public void gradient(double grad) {
        setGrad(grad);
        if (Objects.nonNull(gradient)) gradient.apply(grad, input);
    }

    private double value, grad;
    private Context[] input;
    private Gradient gradient;
}