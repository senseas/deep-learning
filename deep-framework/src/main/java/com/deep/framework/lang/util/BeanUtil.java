package com.deep.framework.lang.util;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.annotation.Operator;

import java.lang.reflect.Method;

public class BeanUtil {

    public static boolean isOperation(Tensor tensor) {
        try {
            Method method = tensor.getClass().getMethod("compute");
            return method.getAnnotation(Operator.class) != null;
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        return false;
    }

    public static boolean isNotOperation(Tensor tensor) {
        try {
            if (tensor == null) return false;
            Method method = tensor.getClass().getMethod("compute");
            return method.getAnnotation(Operator.class) == null;
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        return false;
    }

    public static boolean isTensor(Object o) {
        if (o == null) return false;
        return o.getClass().isArray();
    }

    public static boolean isNotTensor(Object o) {
        if (o == null) return true;
        return !o.getClass().isArray();
    }

    public static boolean isNone(Tensor o) {
        return o.getName().startsWith("None");
    }

    public static boolean isNotNone(Tensor o) {
        return !o.getName().startsWith("None");
    }

    public static boolean startsWithNone(Tensor o) {
        return o.getName().startsWith("None::");
    }

    public static boolean isNoneNode(Tensor o) {
        return o.getName().startsWith("Node") || o.getName().startsWith("None");
    }

    public static void nameNode(Tensor o) {
        o.setName(o.getName().replace("Tensor", "Node"));
    }

}
