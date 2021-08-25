package com.deep.framework.lang.function;

import com.deep.framework.lang.Tenser;

@FunctionalInterface
public interface For2<M> {
    void apply(M l, Tenser<M> n, int i);
}