package com.deep.framework.lang.function;

import com.deep.framework.graph.Node;
import com.deep.framework.graph.None;

@FunctionalInterface
public interface Func3 {
    void apply(Node m, Node n, None o);
}
