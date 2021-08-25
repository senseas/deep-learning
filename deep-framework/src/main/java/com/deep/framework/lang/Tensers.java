package com.deep.framework.lang;

import java.util.concurrent.atomic.AtomicInteger;

import static com.deep.framework.lang.ForEach.farrEach;
import static com.deep.framework.lang.Shape.arrayShapes;
import static com.deep.framework.lang.Shape.size;

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
        farrEach(o, (Double a) -> {
            list[index.getAndIncrement()] = a;
        });
        return new Tenser(list, arrayShapes(o));
    }

}