package com.deep.framework.lang;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.function.*;
import com.deep.framework.lang.util.BeanUtil;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.stream.IntStream;

public class ForEach implements Serializable {

    public static void forEach(int a, Range1 r) {
        IntStream.range(0, a).forEach(i -> r.apply(i));
    }

    public static void forEach(int a, int b, Range2 r) {
        forEach(a, i -> forEach(b, l -> r.apply(i, l)));
    }

    public static void forEach(int a, int b, int c, Range3 r) {
        forEach(a, i -> forEach(b, l -> forEach(c, m -> r.apply(i, l, m))));
    }

    public static void forEach(int a, int b, int c, int e, Range4 r) {
        forEach(a, i -> forEach(b, l -> forEach(c, m -> forEach(e, n -> r.apply(i, l, m, n)))));
    }

    public static Object fill(Object a, Func func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i);
                if (BeanUtil.isNotTensor(m)) {
                    Tensers.set(a, func.apply(m), i);
                } else {
                    fill(m, func);
                }
            });
        }
        return a;
    }

    public static Object fill(Object a, Object b, Func func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i), n = Tensers.get(b, i);
                if (BeanUtil.isNotTensor(m)) {
                    Tensers.set(b, func.apply(m), i);
                } else {
                    fill(m, n, func);
                }
            });
        }
        return b;
    }

    public static <M> void forEach(Object a, Func1<M> func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((M) m);
                } else {
                    forEach(m, func);
                }
            });
        } else {
            func.apply((M) a);
        }
    }

    public static <M, N> void forEach(Object a, Object b, Func2<M, N> func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i), n = Tensers.get(b, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((M) m, (N) n);
                } else {
                    forEach(m, n, func);
                }
            });
        } else {
            func.apply((M) a, (N) b);
        }
    }

    public static <M> void forEach(Object a, Object b, Object c, Func3<M> func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i), n = Tensers.get(b, i), o = Tensers.get(c, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((M) m, (M) n, (M) o);
                } else {
                    forEach(m, n, o, func);
                }
            });
        } else {
            func.apply((M) a, (M) b, (M) c);
        }
    }

    public static <M> void forEach(Object a, For1<M> func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((Tenser<M>) a, i);
                } else {
                    forEach(m, func);
                }
            });
        }
    }

    public static <M> void forEach(Object a, Object b, For2<M> func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i), n = Tensers.get(b, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((M) m, (Tenser<M>) b, i);
                } else {
                    forEach(m, n, func);
                }
            });
        }
    }

    public static <M> void forEach(Object a, Object b, Object c, For3<M> func) {
        if (BeanUtil.isTensor(a)) {
            forEach(Tensers.getLength(a), i -> {
                Object m = Tensers.get(a, i), n = Tensers.get(b, i), o = Tensers.get(c, i);
                if (BeanUtil.isNotTensor(m)) {
                    func.apply((M) m, (M) n, (Tenser<M>) c, i);
                } else {
                    forEach(m, n, o, func);
                }
            });
        }
    }

    public static <M> void arrayEach(Object a, For<M> func) {
        if (BeanUtil.isArray(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i);
                if (BeanUtil.isNotArray(m)) {
                    func.apply((M) a, i);
                } else {
                    arrayEach(m, func);
                }
            });
        }
    }

    public static <M, N> void ferEach(Object a, Object b, For0<M, N> func) {
        if (BeanUtil.isArray(b)) {
            forEach(Array.getLength(b), i -> {
                Object m = Tensers.get(a, i), n = Array.get(b, i);
                if (BeanUtil.isNotArray(n)) {
                    func.apply((M) m, (N[]) b, i);
                } else {
                    ferEach(m, n, func);
                }
            });
        }
    }

    public static Tenser<Tensor> padding(Tenser<Tensor> a, int padding) {
        if (padding == 0) return a;
        int height = a.shape(0), width = a.shape(1);
        int hx = height + 2 * padding, wx = width + 2 * padding;
        Tenser nones = new Tenser(new Tensor[hx * wx], new int[]{hx, wx});

        forEach(padding, wx, (m, n) -> {
            nones.set(new Tensor(0d, false), m, n);
            nones.set(new Tensor(0d, false), m + padding + height, n);
        });

        forEach(hx, padding, (m, n) -> {
            nones.set(new Tensor(0d, false), m, n);
            nones.set(new Tensor(0d, false), m, n + padding + width);
        });

        forEach(height, width, (i, l) -> {
            Object data = a.get(i, l);
            nones.set(data, i + padding, l + padding);
        });
        return nones;
    }

    public static Tensor[][][] paddingTensor(Tensor[][][] a, int padding) {
        if (padding == 0) return a;
        int height = a[0].length, width = a[0][0].length;
        Tensor[][][] tensors = new Tensor[a.length][height + 2 * padding][width + 2 * padding];

        forEach(a.length, padding, tensors[0][0].length, (i, m, n) -> {
            tensors[i][m][n] = new Tensor(0d);
            tensors[i][m + padding + height][n] = new Tensor(0d);
        });

        forEach(a.length, tensors[0].length, padding, (i, m, n) -> {
            tensors[i][m][n] = new Tensor(0d);
            tensors[i][m][n + padding + width] = new Tensor(0d);
        });

        forEach(a.length, height, width, (j, i, l) -> tensors[j][i + padding][l + padding] = a[j][i][l]);
        return tensors;
    }

}