package com.deep.framework.ast;

import java.io.Serializable;
import java.util.List;
import java.util.ListIterator;
import java.util.Objects;

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

    public void reduce(Func2 func) {
        List<Node> list = new NodeList<>(this.list);
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

    public void reduce2(Funcx2 func) {
        NodeIterator<Node> iterator = new NodeIterator(list);
        while (iterator.hasNext()) {
            Node next = iterator.next();
            if (iterator.nextIndex() < list.size() - 1) {
                func.apply(iterator, next, list.get(iterator.nextIndex() + 1));
            } else {
                func.apply(iterator, next, null);
            }
        }
    }

    public void reduce(Func3 func) {
        List<Node> list = new NodeList<>(this.list);
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
    public interface Func2 {
        void apply(List list, Node a, Node b);
    }

    @FunctionalInterface
    public interface Funcx2 {
        void apply(NodeIterator list, Node a, Node b);
    }

    @FunctionalInterface
    public interface Func3 {
        void apply(List list, Node a, Node b, Node c);
    }

    public static class NodeIterator<E> implements ListIterator<E> {
        private List<E> list;
        private E now;

        public NodeIterator(List<E> list) {
            this.list = list;
        }

        @Override
        public boolean hasNext() {
            if (list.isEmpty()) return false;
            int index = nextIndex();
            return index + 1 < list.size();
        }

        @Override
        public E next() {
            int index = nextIndex() + 1;
            if (hasNext()) return now = list.get(index);
            return null;
        }

        @Override
        public int nextIndex() {
            if (Objects.isNull(now)) return -1;
            return list.indexOf(now);
        }

        @Override
        public boolean hasPrevious() {
            if (list.isEmpty()) return false;
            int index = previousIndex();
            return index >= 0;
        }

        @Override
        public E previous() {
            int index = previousIndex();
            if (hasPrevious()) return now = list.get(index);
            return null;
        }

        @Override
        public int previousIndex() {
            int index = nextIndex();
            return index - 1;
        }

        @Override
        public void remove() {
            E o = now;
            next();
            list.remove(o);
            if (list.isEmpty())
                now = null;
        }

        public void remove(E e) {
            now = e;
            if (hasPrevious()) {
                previous();
                list.remove(e);
            } else {
                list.remove(e);
            }
            if (list.isEmpty())
                now = null;
        }

        public void removeAll(E... arr) {
            for (E e : arr) {
                remove(e);
            }
        }

        public void replace(E node, E replaceNode) {
            now = replaceNode;
            int index = list.indexOf(node);
            list.set(index, replaceNode);
        }

        @Override
        public void set(E o) {
            int index = list.indexOf(now);
            list.set(index, o);
        }

        @Override
        public void add(E o) {
            list.set(list.size(), o);
        }
    }

}