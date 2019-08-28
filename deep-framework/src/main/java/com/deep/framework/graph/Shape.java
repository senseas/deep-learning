package com.deep.framework.graph;

import com.deep.framework.bean.None;
import com.deep.framework.lang.ForEach;
import com.deep.framework.lang.function.Fill;
import com.deep.framework.lang.function.Func2;
import com.deep.framework.lang.util.BeanUtil;
import org.apache.commons.math3.random.RandomDataGenerator;

import java.io.*;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

public class Shape extends ForEach {

    static RandomDataGenerator random = new RandomDataGenerator();

    public static <E> E random(int... x) {
        return (E) fill(Array.newInstance(None.class, x), o -> new None(random.nextGaussian(0, 0.1)));
    }

    public static <E> E random(String name, int... x) {
        return (E) fill(Array.newInstance(None.class, x), o -> new None(random.nextGaussian(0, 0.1), name));
    }

    public static <E> E nones(Object a) {
        if (BeanUtil.isTenser(a)) {
            return (E) fill(a, shape(None.class, a), (Fill<Tenser<None>>) o -> o.getOutput());
        } else {
            return (E) ((Tenser) a).getOutput();
        }
    }

    public static <E> E tensers(Object a) {
        if (BeanUtil.isTenser(a)) {
            return (E) fill(a, shape(Tenser.class, a), (Fill<None>) o -> new Tenser(o));
        } else {
            return (E) new Tenser((None) a);
        }
    }

    public static <E> E functions(Object a) {
        return (E) fill(a, a, (Fill<Tenser>) o -> {
            if (BeanUtil.isOperation(o))
                return o;
            return o.getFunction();
        });
    }

    public static <E> E zeros(Object a) {
        return (E) fill(a, o -> new Tenser(0d));
    }

    public static <E> E Nones(Object a) {
        if (BeanUtil.isTenser(a)) {
            return (E) fill(a, shape(None.class, a), o -> new None(0d));
        } else {
            return (E) new None(0d);
        }
    }

    public static Object shape(Class clas, Object a) {
        return Array.newInstance(clas, shapes(a));
    }

    public static <E> int[] shapes(E a, int... list) {
        if (BeanUtil.isTenser(a)) {
            int length = Array.getLength(list);
            list = Arrays.copyOf(list, length + 1);
            Array.set(list, length, Array.getLength(a));
            return shapes(Array.get(a, 0), list);
        }
        return list;
    }

    public static void each(Object a, Object b, Func2 func) {
        if (BeanUtil.isTenser(a)) {
            forEach(Array.getLength(a), i -> {
                Object m = Array.get(a, i), n = Array.get(b, i);
                func.apply(m, n);
            });
        } else {
            func.apply(a, b);
        }
    }

    public static void reshape(Object A, Object B) {
        Queue link = new LinkedList();
        forEach(A, a -> link.add(a));
        forEach(B, (b, i) -> b[i] = link.poll());
    }

    public void saveModel(Object obj, String src) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(src))) {
            out.writeObject(obj);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public <E> E loadModel(String src) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(new File(src)))) {
            Object o = in.readObject();
            return (E) o;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}


