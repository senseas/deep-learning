package com.deep.framework;

import com.alibaba.fastjson.JSONObject;
import com.aparapi.Kernel;
import com.aparapi.Range;
import org.junit.Test;

import java.util.stream.IntStream;

public class AparapiTest {


    @Test
    public void aparapiTest() {
        int a0 = 3, a1 = 2, a2 = 3;
        final double[][] a = new double[a0][a1];
        final double[][] b = new double[a1][a2];
        final double[][][] c = {
            {
                {0, 1},
                {2, 3},
                {4, 5}
            },
            {
                {6, 7},
                {8, 9},
                {10, 11}
            },
            {
                {12, 13},
                {14, 15},
                {16, 17}
            }
        };

        IntStream.range(0, a0).forEach(i -> IntStream.range(0, a1).forEach(l -> a[i][l] = i));
        IntStream.range(0, a1).forEach(i -> IntStream.range(0, a2).forEach(l -> b[i][l] = l));

        final double[][] sum = new double[a0][a2];

        Tenser kernel = new Tenser() {

            public void compute() {
                int i = getGlobalId(0);
                int m = getGlobalId(1);
                int n = getGlobalId(2);
                sum[i][m] += a[i][n] * b[n][m];
            }

        };

        final double[][] sums = new double[a0][a2];
        for (int i = 0; i < a0; i++) {
            for (int m = 0; m < a2; m++) {
                double s = 0;
                for (int n = 0; n < a1; n++) {
                    s += a[i][n] * b[n][m];
                }
                sums[i][m] = s;
            }
        }

        kernel.execute(Range.create3D(a0, a2, a1));

        System.out.println("GPU" + JSONObject.toJSONString(sum));
        System.out.println("CPU" + JSONObject.toJSONString(sums));

        kernel.dispose();
    }


    public abstract class Tenser<T> extends Kernel {

        public void compute() { }

        public void run() { compute(); }

    }

}
