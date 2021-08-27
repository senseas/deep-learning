package com.deep.framework.lang;

import java.lang.reflect.Array;
import java.util.concurrent.atomic.AtomicInteger;

import static com.deep.framework.lang.ForEach.farrEach;
import static com.deep.framework.lang.Shape.*;

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
        Double[] list = new Double[size(o)];
        AtomicInteger index = new AtomicInteger();
        farrEach(o, (a, i) -> {
            list[index.getAndIncrement()] = (Double) Array.get(a, i);
        });
        return new Tenser(list, shapes(o));
    }

    public static <T> T array(Tenser o) {
        Object array = Array.newInstance(Object.class, o.shape);
        AtomicInteger index = new AtomicInteger();
        farrEach(array, (a, i) -> {
            Array.set(a, i, o.data[index.getAndIncrement()]);
        });
        return (T) array;
    }

}