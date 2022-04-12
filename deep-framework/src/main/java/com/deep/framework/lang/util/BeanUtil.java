package com.deep.framework.lang.util;

import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorOperator;
import com.deep.framework.lang.Tenser;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.List;
import java.util.Objects;

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

    public static String tmpl(String content, List<Double> params) {
        String regex = "\\{var\\}";
        for (int i = 0; i <params.size() ; i++) {
            content = content.replaceFirst(regex,"inx[".concat(String.valueOf(i)).concat("]"));
        }
        return content.replaceAll("--","");
    }

}
