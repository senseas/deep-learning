package com.deep.framework.lang.function;

import com.deep.framework.lang.Tenser;

@FunctionalInterface
public interface For3<M> {
    void apply(M l, M m, Tenser<M> n, int i);
}