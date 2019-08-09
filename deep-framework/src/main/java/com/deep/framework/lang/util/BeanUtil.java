package com.deep.framework.lang.util;

import com.deep.framework.bean.Node;
import com.deep.framework.graph.Tenser;
import com.deep.framework.lang.annotation.Operator;

import java.lang.reflect.Method;

public class BeanUtil {

    public static boolean isOperation(Node node) {
        try {
            Method method = node.getClass().getMethod("compute");
            return method.getAnnotation(Operator.class) != null;
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        return false;
    }

    public static boolean isNotOperation(Node node) {
        try {
            Method method = node.getClass().getMethod("compute");
            return method.getAnnotation(Operator.class) == null;
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
        return false;
    }

    public static boolean isTenser(Object o) {
        if (o == null) return false;
        return o.getClass().isArray();
    }

    public static boolean isNotTenser(Object o) {
        if (o == null) return true;
        return !o.getClass().isArray();
    }

    public static boolean isNone(Tenser o) {
        return o.getName().startsWith("None");
    }

    public static boolean isNotNone(Tenser o) {
        return !o.getName().startsWith("None");
    }

    public static boolean startsWithNone(Tenser o) {
        return o.getName().startsWith("None::");
    }
}
