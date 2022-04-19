package com.deep.framework.lang.util;

import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorOperator;
import com.deep.framework.lang.Tenser;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

public class BeanUtil {

    static RandomDataGenerator random = new RandomDataGenerator();

    public static boolean isOperation(Tensor tensor) {
        return tensor instanceof TensorOperator;
    }

    public static boolean isNotOperation(Tensor tensor) {
        return !(tensor instanceof TensorOperator);
    }

    public static boolean isTensor(Object o) {
        if (Objects.isNull(o)) return false;
        return o instanceof Tenser;
    }

    public static boolean isNotTensor(Object o) {
        if (Objects.isNull(o)) return true;
        return !(o instanceof Tenser);
    }

    public static boolean isArray(Object o) {
        if (Objects.isNull(o)) return false;
        return o.getClass().isArray();
    }

    public static boolean isNotArray(Object o) {
        if (Objects.isNull(o)) return true;
        return !(o.getClass().isArray());
    }

    public static boolean isNone(Tensor o) {
        return o.getName().startsWith("None");
    }

    public static boolean isNotNone(Tensor o) {
        return !o.getName().startsWith("None");
    }

    public static boolean isFunction(Tensor o) {
        return o.getName().equals("Function");
    }

    public static double toDouble(double o) {
        if (Double.isNaN(o)) return 0;
        if (Double.isInfinite(o)) return 0.9;
        return o;
    }

    public static String getCode(String name, String content) {
        AtomicInteger index = new AtomicInteger();
        StringBuilder code = new StringBuilder("extern \"C\" __global__ void ")
        .append(name).append("(double* inx , double* out){ out[0] = ");
        content.chars().mapToObj(a -> String.valueOf((char) a)).reduce((a, b) -> {
            if (a.equals("{")) {
                return a.concat(b);
            }
            if (b.equals("{")) {
                code.append(a);
                return "{";
            }
            if (a.concat(b).equals("{var}")) {
                code.append("inx[").append(index.toString()).append("]");
                index.incrementAndGet();
                return "";
            }
            if (a.isEmpty()) {
                code.append(b);
                return "";
            }
            return a.concat(b);
        });
        return code.append(";}").toString()
            .replaceAll("--", "");
    }

}
