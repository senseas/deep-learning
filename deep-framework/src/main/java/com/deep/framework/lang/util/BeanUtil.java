package com.deep.framework.lang.util;

import com.deep.framework.lang.Tenser;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.util.Objects;

public class BeanUtil {

    static RandomDataGenerator random = new RandomDataGenerator();

    public static boolean isTenser(Object o) {
        if (Objects.isNull(o)) return false;
        return o instanceof Tenser;
    }

    public static boolean isNotTenser(Object o) {
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

    public static double toDouble(double o) {
        if (Double.isNaN(o)) return 0;
        if (Double.isInfinite(o)) return 0.9;
        return o;
    }

}
