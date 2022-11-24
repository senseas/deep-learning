package com.deep.framework.ast;

import com.deep.framework.ast.lexer.TokenType;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Stream implements Serializable {
    private final List<Node> list;

    private Stream(List list) {
        this.list = list;
    }

    public static Stream of(List list) {
        return new Stream(list);
    }

    public static <E> NodeList<E> of(E... list) {
        return new NodeList<E>(list);
    }

    public void contains(TokenType type, Func1 func) {
        func.apply(list.stream().filter(a -> a.equals(type)).findFirst().get());
    }

    public void reduce(Func2 func) {
        List<Node> list = new ArrayList(this.list);
        while (0 < list.size()) {
            Node o = list.get(0);
            list.remove(o);
            if (0 < list.size()) {
                func.apply(list, o, list.get(0));
            } else {
                func.apply(list, o, null);
            }
        }
    }

    public void reduce(Func3 func) {
        List<Node> list = new ArrayList(this.list);
        while (0 < list.size()) {
            Node o = list.get(0);
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
    public interface Func1 {
        void apply(Node a);
    }

    @FunctionalInterface
    public interface Func2 {
        void apply(List list, Node a, Node b);
    }

    @FunctionalInterface
    public interface Func3 {
        void apply(List list, Node a, Node b, Node c);
    }

}

