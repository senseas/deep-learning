package com.deep.framework.lang.flow;

@FunctionalInterface
public interface Gradient {
    void apply(double grad, Context... input);
}