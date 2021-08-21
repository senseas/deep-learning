package com.deep.framework;

import com.deep.framework.lang.Tenser;
import org.junit.Test;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

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
                {1d, 2d, 3d},
                {5d, 4d, 6d},
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

        Array a = array.get(0, 0);
        a.forEach(i -> System.out.println(a.get(i)));
    }

    @Test
    public void tenserTest() {
        Tenser<Double> tenser = new Tenser(new Double[3 * 2 * 3], new int[]{3, 2, 3});
        long s = System.currentTimeMillis();
        AtomicReference<Double> cc = new AtomicReference<>(0d);
        IntStream.range(0, 3).forEach(l -> {
            IntStream.range(0, 2).forEach(m -> {
                IntStream.range(0, 3).forEach(n -> {
                    cc.set(cc.get() + 1);
                    tenser.set(cc.get(), l, m, n);
                });
            });
        });

        Tenser tensers = tenser.get(2, 1);
        tenser.set(new Double[]{1d, 1d, 1d}, 2, 1);

        IntStream.range(0, 3).forEach(l -> {
            IntStream.range(0, 2).forEach(m -> {
                IntStream.range(0, 3).forEach(n -> {
                    Object o = tenser.get(l, m, n);
                    System.out.println(o);
                });
            });
        });
        System.out.println((System.currentTimeMillis() - s) / 1000d);

        double[][][] data1 = new double[800][200][300];
        s = System.currentTimeMillis();
        IntStream.range(0, 800).forEach(l -> {
            IntStream.range(0, 200).forEach(m -> {
                IntStream.range(0, 300).forEach(n -> {
                    double v = data1[l][m][n];
                });
            });
        });
        System.out.println((System.currentTimeMillis() - s) / 1000d);

        double[] data2 = new double[800 * 200 * 300];
        s = System.currentTimeMillis();
        IntStream.range(0, 800).forEach(l -> {
            int x = l * 200 * 300;
            IntStream.range(0, 200).forEach(m -> {
                int y = m * 300;
                IntStream.range(0, 300).forEach(n -> {
                    double v = data2[x + y + n];
                });
            });
        });
        System.out.println((System.currentTimeMillis() - s) / 1000d);
    }
}
