package com.deep.framework.lang;

import static com.deep.framework.lang.Shape.shapes;

public class Tensers {

    public static <T> int getLength(T o) {
        Tenser a = (Tenser) o;
        return a.getLength();
    }

    public static <T> T get(Object o, int... index) {
        Tenser<T> a = (Tenser) o;
        return a.get(index);
    }

    public static void set(Object o, Object value, int... index) {
        Tenser a = (Tenser) o;
        a.set(value, index);
    }

    public static Tenser tenser(Object o) {
        int[] shape = shapes(o);
        return new Tenser(o, shape);
    }

}