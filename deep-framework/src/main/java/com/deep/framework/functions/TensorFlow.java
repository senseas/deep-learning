package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import static com.deep.framework.core.TensorExecutor.eps;
import static com.deep.framework.lang.ForEach.forEach;

public class TensorFlow implements Operator {

    public Tensor layerNormal(Tensor... input) {
        return new TensorFunction("LayerNormal", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1), C = getInput(2);
                Tensor tensor = Tensor(A), mean = mean(tensor), std = standard(tensor, mean);
                Tenser<Tensor> D = zeroTensors(A.shape);
                Tensor pow = pow(add(std, cons(eps)), cons(0.5));
                forEach(A, B, C, D, (Tensor a, Tensor b, Tensor c) -> {
                    return add(mul(b, div(minus(a, mean), pow)), c);
                });
                return D;
            }

        };
    }

    public Tensor standard(Tensor... input) {
        return new TensorFunction("Standard", new int[]{1}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> inx = getInput(0), iny = getInput(1);
                Tensor mean = iny.one(), cons = cons(2);
                Tenser<Tensor> pows = zeroTensors(inx.shape);
                forEach(inx, pows, (Tensor a) -> pow(minus(a, mean), cons));
                return new Tenser<>(mean(Tensor(pows)));
            }

        };
    }


    public Tensor mean(Tensor... input) {
        return new TensorFunction("Mean", new int[]{1}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> inx = getInput(0);
                return new Tenser<>(div(sum(Tensor(inx)), cons(inx.size())));
            }

        };
    }

    public Tensor sigmoid(Tensor input) {
        return new TensorFunction("Sigmoid", new int[]{1}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> input = getInput(0);
                return new Tenser<>(div(cons(1d), add(cons(1d), exp(minus(input.one())))));
            }

        };
    }

    public static void main(String[] args) {
        TensorFlow tf = new TensorFlow();
        Tensor data1 = new Tensor(new int[]{3, 1});
        Tensor data2 = new Tensor(new int[]{3, 1});
        Tensor data3 = new Tensor(new int[]{3, 1});
        Tensor layerNormal = tf.layerNormal(data1, data2, data3);

        layerNormal.forward();
        forEach(layerNormal.getOutput(), (Tensor out) -> out.setGrad(tf.cons(1)));
        layerNormal.backward();
        Tensor grad = layerNormal.getInput()[0].getOutput().one().grad;
        //grad.forward();
        grad.reducer();
    }
}