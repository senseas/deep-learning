package com.deep.framework;

import com.aparapi.Kernel;
import com.aparapi.Range;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

@Slf4j
public class AparapiTest {

    @Test
    public void arrayTest() {
        int a0 = 3, a1 = 2, a2 = 3;
        final float[][][] c = new float[a0][a2][a1];
        final float[][] a = new float[a0][a1];
        final float[][] b = new float[a1][a2];

        for (int x = 0; x < a0; x++) {
            for (int y = 0; y < a1; y++) {
                a[x][y] = x;
            }
        }

        for (int x = 0; x < a1; x++) {
            for (int y = 0; y < a2; y++) {
                b[x][y] = y;
            }
        }

        final float[][] sum = new float[a0][a2];

        Kernel kernel = new Kernel() {
            public void run() {
                int x = getGlobalId(0);
                int y = getGlobalId(1);
                int z = getGlobalId(2);
                c[x][y][z] += a[x][z] * b[z][y];
                sum[x][y]  += c[x][y][z];
            }
        };

        kernel.execute(Range.create3D(a0, a2, a1));

        kernel.dispose();
    }
}

