package com.deep.framework.ast;

import java.util.*;
import java.util.stream.IntStream;

public class NodeList<E> extends AbstractList<E> {
    private int size;
    private int capacity = 10;
    private E[] array;

    public NodeList() {
        array = (E[]) new Object[capacity];
    }

    @Override
    public int size() {
        return size;
    }

    public E get(int index) {
        check(index);
        return array[index];
    }

    public E set(int index, E e) {
        check(index);
        return array[index] = e;
    }

    private void check(int index) {
        if (index > size) throw new ArrayIndexOutOfBoundsException(index);
    }

    @Override
    public void add(int index, E e) {
        check(index);
        if (size < capacity) {
            if (size == index) {
                array[size++] = e;
            } else {
                Object[] newArray = new Object[capacity];
                System.arraycopy(array, 0, newArray, 0, index);
                newArray[index] = e;
                System.arraycopy(array, index, newArray, index + 1, size - index);
                size++;
                array = (E[]) newArray;
            }
        } else {
            capacity += 10;
            Object[] newArray = new Object[capacity];
            newArray[index] = e;
            System.arraycopy(array, 0, newArray, 0, index);
            size++;
            array = (E[]) newArray;
        }
    }

    @Override
    public boolean remove(Object o) {
        Iterator<E> it = iterator();
        if (o == null) {
            while (it.hasNext()) {
                if (it.next() == null) {
                    it.remove();
                    return true;
                }
            }
        } else {
            while (it.hasNext()) {
                if (o == it.next()) {
                    it.remove();
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        Objects.requireNonNull(c);
        boolean modified = false;
        Iterator<?> it = iterator();
        while (it.hasNext()) {
            if (c.contains(it.next())) {
                it.remove();
                modified = true;
            }
        }
        return modified;
    }

    @Override
    public E remove(int index) {
        check(index);
        int l = 0;
        E c = null;
        Object[] newArray = new Object[capacity];
        for (int i = 0; i < size; i++) {
            if (i == index) {
                c = array[i];
            } else {
                newArray[l++] = array[i];
            }
        }
        size = l;
        array = (E[]) newArray;
        return c;
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        Objects.requireNonNull(c);
        boolean modified = false;
        Iterator<E> it = iterator();
        while (it.hasNext()) {
            if (!c.contains(it.next())) {
                it.remove();
                modified = true;
            }
        }
        return modified;
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        for (Object e : c) {
            if (!contains(e)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean contains(Object o) {
        Iterator<E> it = iterator();
        if (o == null) {
            while (it.hasNext()) {
                if (it.next() == null) {
                    return true;
                }
            }
        } else {
            while (it.hasNext()) {
                if (o == it.next()) {
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public int indexOf(Object o) {
        ListIterator<E> it = listIterator();
        if (o == null) {
            while (it.hasNext()) {
                if (it.next() == null) {
                    return it.previousIndex();
                }
            }
        } else {
            while (it.hasNext()) {
                if (o == it.next()) {
                    return it.previousIndex();
                }
            }
        }
        return -1;
    }

    @Override
    public int lastIndexOf(Object o) {
        ListIterator<E> it = listIterator(size());
        if (o == null) {
            while (it.hasPrevious()) {
                if (it.previous() == null) {
                    return it.nextIndex();
                }
            }
        } else {
            while (it.hasPrevious()) {
                if (o == it.previous()) {
                    return it.nextIndex();
                }
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        List<Object> objects = new NodeList<Object>();
        IntStream.range(0, 15).forEach(objects::add);
        objects.add(0, 100);
        objects.remove(0);
        objects.remove((Integer) 5);
        System.out.println(1);
    }

}