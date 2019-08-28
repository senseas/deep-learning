package com.deep.framework.lang.function;

import com.deep.framework.graph.Tensor;

@FunctionalInterface
public interface For3 {
    void apply(Tensor l, Tensor m, Tensor[] n, int i);
}