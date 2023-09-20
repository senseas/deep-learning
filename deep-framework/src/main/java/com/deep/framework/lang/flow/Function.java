package com.deep.framework.lang.flow;

public interface Function extends Operator {
    default void compute() {
    }

    default void gradient(double grad) {
    }
}