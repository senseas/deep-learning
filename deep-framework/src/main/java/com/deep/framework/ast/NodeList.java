package com.deep.framework.ast;

import java.util.*;
import java.util.function.UnaryOperator;
import java.util.stream.IntStream;

public class NodeList<E> implements List<E> {
    private int size;
    private int capacity;
    private E[] data;

    public NodeList() {
        data = (E[]) new Object[capacity];
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean isEmpty() {
        return size == 0;
    }

    @Override
    public boolean add(E c) {
        if (size < capacity) {
            data[size++] = c;
            return true;
        }

        capacity += 10;
        E[] d = (E[]) new Object[capacity];
        size = 0;
        for (E e : data) {
            d[size++] = e;
        }
        d[size++] = c;
        data = d;
        return true;
    }

    @Override
    public boolean remove(Object o) {
        if (contains(o)) {
            E[] d = (E[]) new Object[capacity];
            size = 0;
            for (E a : data) {
                if (a != o) {
                    d[size++] = a;
                }
            }
            data = d;
            return true;
        }
        return false;
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        if (size + c.size() < capacity) {
            for (E e : c) {
                data[size++] = e;
            }
        } else {
            capacity += Math.ceil(c.size() / 10.0) * 10;
            E[] d = (E[]) new Object[capacity];
            size = 0;
            for (E e : data) {
                d[size++] = e;
            }
            for (E e : c) {
                d[size++] = e;
            }
            data = d;
        }
        return true;
    }

    @Override
    public boolean addAll(int index, Collection<? extends E> c) {
        return false;
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        for (Object e : c) {
            remove(e);
        }
        return true;
    }


    public E get(int index) {
        if (index < size) {
            return data[index];
        }
        throw new ArrayIndexOutOfBoundsException(index);
    }

    public E set(int index, E e) {
        if (index < size) {
            data[index] = e;
        }
        throw new ArrayIndexOutOfBoundsException(index);
    }

    @Override
    public void add(int index, E element) {

    }

    public E remove(int index) {
        if (index < size) {
            for (int i = 0; i < size; i++) {
                if (i == index) {
                    E e = data[i];
                    remove(e);
                    return e;
                }
            }
        }
        throw new ArrayIndexOutOfBoundsException(index);
    }

    @Override
    public boolean contains(Object o) {
        for (E e : data) {
            if (e == o) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        for (E e : data) {
            for (Object o : c) {
                if (e != o) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        for (E e : data) {
            for (Object o : c) {
                if (e != o) {
                    remove(e);
                }
            }
        }
        return false;
    }

    @Override
    public void replaceAll(UnaryOperator<E> operator) {
        List.super.replaceAll(operator);
    }

    @Override
    public void sort(Comparator<? super E> c) {
        List.super.sort(c);
    }

    @Override
    public void clear() {
        size = 0;
        data = (E[]) new Object[10];
    }

    @Override
    public Iterator<E> iterator() {
        return this.iterator();
    }

    @Override
    public Object[] toArray() {
        Object[] d = new Object[size];
        System.arraycopy(data, 0, d, 0, size);
        return d;
    }

    @Override
    public <T> T[] toArray(T[] a) {
        System.arraycopy(data, 0, a, 0, size);
        return a;
    }

    public int indexOf(Object o) {
        for (int i = 0; i < size; i++) {
            if (data[i] == o) {
                return i;
            }
        }
        return -1;
    }

    @Override
    public int lastIndexOf(Object o) {
        return 0;
    }

    @Override
    public ListIterator<E> listIterator() {
        return null;
    }

    @Override
    public ListIterator<E> listIterator(int index) {
        return null;
    }

    @Override
    public List<E> subList(int fromIndex, int toIndex) {
        return null;
    }

    @Override
    public Spliterator<E> spliterator() {
        return List.super.spliterator();
    }

    public static void main(String[] args) {
        List<Object> objects = new NodeList<Object>();
        IntStream.range(0, 15).forEach(i -> objects.add(i));
        IntStream.range(0, 15).forEach(i -> objects.remove(objects.get(i)));
        System.out.println(1);
    }

}