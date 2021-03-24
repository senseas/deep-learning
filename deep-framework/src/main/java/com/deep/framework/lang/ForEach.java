package com.deep.framework.lang;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.function.*;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.stream.IntStream;

public class ForEach implements Serializable {

    public static void farEach(int a, Range1 r) {
        IntStream.range(0, a).parallel().forEach(i -> r.apply(i));
    }

    public static void forEach(int a, Range1 r) {
        IntStream.range(0, a).forEach(i -> r.apply(i));
    }

    public static void forEach(int a, int b, Range2 r) {
        forEach(a, i -> forEach(b, l -> r.apply(i, l)));
    }

    public static void forEach(int a, int b, int c, Range3 r) {
        forEach(a, i -> forEach(b, l -> forEach(c, m -> r.apply(i, l, m))));
    }

    public static void farEach(int a, int b, int c, Range3 r) {
        farEach(a, i -> farEach(b, l -> farEach(c, m -> r.apply(i, l, m))));
    }

    public static void forEach(int a, int b, int c, int e, Range4 r) {
        forEach(a, i -> forEach(b, l -> forEach(c, m -> forEach(e, n -> r.apply(i, l, m, n)))));
    }

    public static void farEach(int a, int b, int c, int e, Range4 r) {
        farEach(a, i -> farEach(b, l -> farEach(c, m -> farEach(e, n -> r.apply(i, l, m, n)))));
    }

    public static Object fill(Object a, Fill func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i);
                if (BeanUtil.isNotTensor(m)) {
                    Array.set(a, i, func.apply(m));
                } else {
                    fill(m, func);
                }
            });
        }
        return a;
    }

    public static Object fill(Object a, Object b, Fill func) {
        if (BeanUtil.isTensor(a)) {
            farEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i), n = Array.get(b, i);
                if (BeanUtil.isNotTensor(m)) {
                    Array.set(b, i, func.apply(m));
                } else {
                    fill(m, n, func);
                }
            });
        }
        return b;
    }

    public static void forEach(Object a, Func1 func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply(m);
                } else {
                    forEach(m, func);
                }
            });
        } else {
            func.apply(a);
        }
    }

    public static void farEach(Object a, Func1 func) {
        if (BeanUtil.isTensor(a)) {
            farEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply(m);
                } else {
                    farEach(m, func);
                }
            });
        } else {
            func.apply(a);
        }
    }

    public static void forEach(Object a, Object b, Func2 func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i), n = Array.get(b, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply(m, n);
                } else {
                    forEach(m, n, func);
                }
            });
        } else {
            func.apply(a, b);
        }
    }

    public static void farEach(Object a, Object b, Func2 func) {
        if (BeanUtil.isTensor(a)) {
            farEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i), n = Array.get(b, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply(m, n);
                } else {
                    farEach(m, n, func);
                }
            });
        } else {
            func.apply(a, b);
        }
    }

    public static void forEach(Object a, For1 func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((Object[]) a, i);
                } else {
                    forEach(m, func);
                }
            });
        }
    }

    public static void forEach(Object a, Object b, For2 func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i), n = Array.get(b, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((Tensor) m, (Tensor[]) b, i);
                } else {
                    forEach(m, n, func);
                }
            });
        }
    }

    public static void forEach(Object a, Object b, Object c, For3 func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i), n = Array.get(b, i), o = Array.get(c, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((Tensor) m, (Tensor) n, (Tensor[]) c, i);
                } else {
                    forEach(m, n, o, func);
                }
            });
        }
    }
}


