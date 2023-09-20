package com.deep.framework.lang.flow;

public interface Application {

    default void compute() { }

    void gradient(double grad);

}