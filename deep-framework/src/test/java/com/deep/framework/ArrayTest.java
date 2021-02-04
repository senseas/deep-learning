package com.deep.framework;

import org.junit.Test;

import java.util.Arrays;
import java.util.function.IntConsumer;

public class ArrayTest {

    public static class Array<T> {
        private final int[] shape;
        private final T[] data;

        public Array(int[] shape, T[] data) {
            this.data = data;
            this.shape = shape;
        }

        public <E> E get(Integer index) {
            if (shape.length == 1) return (E) data[index];
            int row = data.length / shape[0];
            int[] d = Arrays.copyOfRange(shape, 1, shape.length);
            T[] b = Arrays.copyOfRange(data, index * row, index * row + row);
            return (E) new Array(d, b);
        }

        public <E> E get(Integer... index) {
            Object a = this.get(index[0]);
            if (index.length == 1) return (E) a;
            index = Arrays.copyOfRange(index, 1, index.length);
            Array b = (Array) a;
            return (E) b.get(index);
        }

        public void forEach(IntConsumer a) {
            for (int i = 0; i < shape[0]; i++)
                a.accept(i);
        }
    }

    @Test
    public void arrayTest() {
        Double[][][] cc = new Double[][][]{
            {
                {1d, 2d, 5d},
                {6d, 3d, 4d},
            },
            {
                {7d, 8d, 9d},
                {10d, 11d, 12d}
            },
            {
                {13d, 14d, 15d},
                {16d, 17d, 18d},
            }
        };

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

        Array a = array.get(2, 0);
        a.forEach(i -> System.out.println(a.get(i)));
    }
}
