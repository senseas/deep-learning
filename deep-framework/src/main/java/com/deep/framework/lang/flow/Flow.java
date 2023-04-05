package com.deep.framework.lang.flow;

public interface Flow {

    default void compute() { }

    void gradient();

}