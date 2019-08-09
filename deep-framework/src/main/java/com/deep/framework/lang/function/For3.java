package com.deep.framework.lang.function;

import com.deep.framework.graph.Tenser;

@FunctionalInterface
public interface For3 {
    void apply(Tenser l, Tenser m, Tenser[] n, int i);
}