package com.deep.framework.lang.function;

@FunctionalInterface
public interface Func<N> {
    Object apply(N o);
}
