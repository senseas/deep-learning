package com.deep.framework.lang;

import java.util.stream.IntStream;

import static com.deep.framework.lang.TensorUtil.zeros;


public class TensorFlow {

    public Tensor matmul(Tensor... input) {

        return new Tensor("matmul", input) {

            public Object compute(TensorContext context) {
                Tensor inx = input[0], iny = input[1];
                int x = inx.shape[0], y = iny.shape[1], h = inx.shape[1];
                this.shape = new int[]{x, y};
                return context.setArgs(x, y, h).compute(x, y);
            }

            public void gradient(TensorContext context) {
                Tensor inx = input[0], iny = input[1];
                int x = inx.shape[0], y = iny.shape[1], h = inx.shape[1];
                context.gradient(x, y);
            }

        };

    }

    public Tensor matmulx(Tensor... input) {

        return new Tensor("matmulx", input) {

            public Object compute(TensorContext context) {
                Tensor inx = input[0], iny = input[1];
                int x = inx.shape[0], h = inx.shape[1], y = iny.shape[1];
                zeros(this, x, y);
                context.setArgs(inx, iny, this, x, h, y, x, y);
                return context.compute(x, y);
            }

        };

    }

    public static void main(String[] args) {
        TensorFlow tf = new TensorFlow();
        TensorExecutor executor = new TensorExecutor();
        long start = System.currentTimeMillis();
        Tensor A = new Tensor(new int[]{2, 3});
        Tensor B = new Tensor(new int[]{3, 4});
        Tensor C = tf.matmul(A, B);

        C.compute(executor.createContext(C));
        System.out.println((System.currentTimeMillis() - start) / 1000);
        IntStream.range(0, C.output.length).forEach(i -> System.out.println(C.output[i]));

//        start = System.currentTimeMillis();
//        Tensor M = new Tensor(new int[]{3000, 2000});
//        Tensor N = new Tensor(new int[]{2000, 3000});
//        Tensor O = tf.matmulx(M, N);
//        O.compute(executor.createContext(O.name));
//        System.out.println((System.currentTimeMillis() - start) / 1000);
//        FloatBuffer buffero = (FloatBuffer) O.buffer.getBuffer();
//        IntStream.range(0, buffer.capacity()).forEach(i -> System.out.println(buffero.get(i)));
    }
}
