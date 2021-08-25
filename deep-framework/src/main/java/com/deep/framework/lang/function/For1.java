package com.deep.framework.lang.function;

import com.deep.framework.lang.Tenser;

@FunctionalInterface
public interface For1<M> {
    void apply(Tenser<M> m, int i);
}

