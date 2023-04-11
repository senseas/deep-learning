package com.deep.framework.lang.flow;

@FunctionalInterface
public interface Gradient {
    AppContext apply(double grad);
}