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
        int[] shape = shapes(o);
        AtomicInteger index = new AtomicInteger();
        Object array = Array.newInstance(getArrayDeepClass(o), size(shape));
        farrEach(o, (a, i) -> Array.set(array, index.getAndIncrement(), Array.get(a, i)));
        return new Tenser((Object[]) array, shape);
    }

    public static <T> T array(Tenser o) {
        AtomicInteger index = new AtomicInteger();
        Object array = Array.newInstance(getTenserDeepClass(o), o.shape);
        farrEach(array, (a, i) -> Array.set(a, i, o.data[index.getAndIncrement()]));
        return (T) array;
    }

}