package com.deep.framework.ast;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Stream implements Serializable {
    private final List list;

    private Stream(List list) {
        this.list = list;
    }

    public static Stream of(List list) {
        return new Stream(list);
    }

    public void reduce(Func2 func) {
        List list = new ArrayList(this.list);
        while (0 < list.size()) {
            Object o = list.get(0);
            list.remove(o);
            if (0 < list.size()) {
                func.apply(list, o, list.get(0));
            } else {
                func.apply(list, o, null);
            }
        }
    }

    public void reduce(Func3 func) {
        List list = new ArrayList(this.list);
        while (0 < list.size()) {
            Object o = list.get(0);
            list.remove(o);
            if (1 < list.size()) {
                func.apply(list, o, list.get(0), list.get(1));
            } else if (0 < list.size()) {
                func.apply(list, o, list.get(0), null);
            } else {
                func.apply(list, o, null, null);
            }
        }
    }

    @FunctionalInterface
    public interface Func2 {
        void apply(List list, Object a, Object b);
    }

    @FunctionalInterface
    public interface Func3 {
        void apply(List list, Object a, Object b, Object c);
    }

}

