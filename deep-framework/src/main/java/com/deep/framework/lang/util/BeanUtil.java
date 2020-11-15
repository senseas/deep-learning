package com.deep.framework.lang.util;

import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorOparetor;

import java.util.Objects;

public class BeanUtil {

    public static boolean isOperation(Tensor tensor) {
        return tensor instanceof TensorOparetor;
    }

    public static boolean isNotOperation(Tensor tensor) {
        return !(tensor instanceof TensorOparetor);
    }

    public static boolean isTensor(Object o) {
        if (Objects.isNull(o)) return false;
        return o.getClass().isArray();
    }

    public static boolean isNotTensor(Object o) {
        if (Objects.isNull(o)) return true;
        return !o.getClass().isArray();
    }

    public static boolean isNone(Tensor o) {
        return o.getName().startsWith("None");
    }

    public static boolean isNotNone(Tensor o) {
        return !o.getName().startsWith("None");
    }

    public static boolean isFunction(Tensor o) {
        return o.getName().startsWith("Tensor");
    }

    public static boolean startsWithNone(Tensor o) {
        return o.getName().startsWith("None::");
    }

    public static void nameNode(Tensor o) {
        o.setName(o.getName().replace("Tensor", "Node"));
    }

}
