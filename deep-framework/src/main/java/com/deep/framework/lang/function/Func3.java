package com.deep.framework.lang.function;

import com.deep.framework.bean.Node;
import com.deep.framework.bean.None;

@FunctionalInterface
public interface Func3 {
    void apply(Node m, Node n, None o);
}
