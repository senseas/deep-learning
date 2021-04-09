package com.deep.framework.lang.function;

@FunctionalInterface
public interface Fill<N> {
    Object apply(N o);
}
