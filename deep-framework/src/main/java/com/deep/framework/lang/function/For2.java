package com.deep.framework.lang.function;

import com.deep.framework.graph.Tensor;

@FunctionalInterface
public interface For2 {
    void apply(Tensor l, Tensor[] n, int i);
}

