package com.deep.framework.lang.function;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;

@FunctionalInterface
public interface Func3 {
    void apply(Tensor m, Tensor n, None o);
}
