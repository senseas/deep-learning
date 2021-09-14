package com.deep.framework.lang.function;

@FunctionalInterface
public interface Func2<M,N> {
    void apply(M m, N n);
}
