package com.deep.framework.lang;

import java.nio.FloatBuffer;

import static com.deep.framework.lang.TensorUtil.zeros;


public class TensorFlow {

    public Tensor matmul(Tensor... input) {

        return new Tensor("matmul", input) {

            public void compute(TensorContext context) {
                Tensor inx = input[0], iny = input[1];
                int x = inx.shape[0], h = inx.shape[1], y = iny.shape[1];
                zeros(this, x, y);
                context.setArgs(inx, iny, this, x, h, y);
                context.excute(x, y);
            }

        };

    }

    public Tensor matmulx(Tensor... input) {

        return new Tensor("matmulx", input) {

            public void compute(TensorContext context) {
                Tensor inx = input[0], iny = input[1];
                int x = inx.shape[0], h = inx.shape[1], y = iny.shape[1];
                zeros(this, x, y);
                context.setArgs(inx, iny, this, x, h, y, x, y);
                context.excute(x, y);
            }

        };

    }

    public static void main(String[] args) {
        TensorFlow tf = new TensorFlow();
        TensorExecutor executor = new TensorExecutor();
        long start = System.currentTimeMillis();
        Tensor A = new Tensor(new int[]{3000, 2000});
        Tensor B = new Tensor(new int[]{2000, 3000});
        Tensor C = tf.matmul(A, B);

        C.compute(executor.createContext(C.name));
        System.out.println((System.currentTimeMillis() - start) / 1000);

        FloatBuffer buffer = (FloatBuffer) C.buffer.getBuffer();
        //IntStream.range(0, buffer.capacity()).forEach(i -> System.out.println(buffer.get(i)));

        //----------------------------------------------
        start = System.currentTimeMillis();
        Tensor M = new Tensor(new int[]{3000, 2000});
        Tensor N = new Tensor(new int[]{2000, 3000});
        Tensor O = tf.matmulx(M, N);

        O.compute(executor.createContext(O.name));
        System.out.println((System.currentTimeMillis() - start) / 1000);
        FloatBuffer buffero = (FloatBuffer) O.buffer.getBuffer();
        //IntStream.range(0, buffer.capacity()).forEach(i -> System.out.println(buffero.get(i)));
    }
}
