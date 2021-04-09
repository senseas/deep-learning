package com.deep.framework;

import org.junit.Test;

import java.util.function.IntConsumer;

public class ArrayTestx {

    public class Array<T> {
        private int[] shape;
        private T[] data;

        private int[] d;

        public Array(int[] shape, T[] data) {
            this.shape = shape;
            this.data = data;
            this.shape();
        }

        public T get(Integer... index) {
            int idx = 0, next;
            for (int i = 0; i < index.length - 1; i++) {
                next = 1;
                for (int l = i + 1; l < shape.length; l++) {
                    next = next * shape[l];
                }
                idx = idx + index[i] * next;
            }
            return data[idx + index.length - 1];
        }

        public void shape() {
            d = new int[shape.length - 1];
            for (int i = 0; i < shape.length - 1; i++)
                d[i] = shape[1 + i];
        }

        public void forEach(IntConsumer a) {
            for (int i = 0; i < shape[0]; i++)
                a.accept(i);
        }
    }

    @Test
    public void arrayTest() {
        Array<Double> array = new Array(
            new int[]{3, 2, 3},
            new Double[]{
                1d, 2d, 3d,
                4d, 5d, 6d,

                7d, 8d, 9d,
                10d, 11d, 12d,

                13d, 14d, 15d,
                16d, 17d, 18d
            }
        );

        Double m = array.get(2, 1, 2);
        System.out.println(m);
        /*array.forEach(x -> {
            Array<Double> a = array.get(x);
            a.forEach(y -> {
                Array<Double> b = a.get(y);
                b.forEach(z -> {
                    Double c = b.get(z);
                    System.out.println(c);
                });
            });
        });*/

    }
}
