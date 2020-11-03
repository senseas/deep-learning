package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

public class Builder extends Shape {

    public static <E> E getOutput(Object a) {
        if (BeanUtil.isTensor(a)) {
            return (E) fill(a, shape(None.class, a), b -> {
                Tensor o = (Tensor) b;
                return o.getOutput();
            });
        } else {
            Tensor o = (Tensor) a;
            return (E) o.getOutput();
        }
    }

}
