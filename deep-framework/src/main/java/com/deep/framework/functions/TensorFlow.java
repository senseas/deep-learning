package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import static com.deep.framework.core.TensorExecutor.eps;
import static com.deep.framework.lang.ForEach.forEach;
import static com.deep.framework.lang.Shape.size;

public class TensorFlow implements Operator {

    public Tensor layerNormal(Tensor... input) {
        return new TensorFunction("LayerNormal", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1), C = getInput(2);
                Tensor mean = mean(getInput()[0]), std = standard(getInput()[0], mean);
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
                Tenser<Tensor> inx = getInput(0);
                Tensor mean = getInput()[1], cons = cons(2);
                Tenser<Tensor> pows = zeroTensors(inx.shape);
                forEach(inx, pows, (Tensor a) -> pow(minus(a, mean), cons));
                return new Tenser<>(mean(Tensor(pows)));
            }

        };
    }

    public Tensor mean(Tensor... input) {
        return new TensorFunction("Mean", new int[]{1}, input) {

            public Tenser<Tensor> compute() {
                Tensor inx = getInput()[0];
                return new Tenser<>(div(sum(inx), cons(size(inx.shape))));
            }

        };
    }

    public Tensor sigmoid(Tensor input) {
        return new TensorFunction("Sigmoid", new int[]{1}, input) {

            public Tenser<Tensor> compute() {
                Tensor input = getInput()[0];
                return new Tenser<>(div(cons(1d), add(cons(1d), exp(minus(input)))));
            }

        };
    }

    public Tensor sigmoidCross(Tensor... input) {
        return new TensorFunction("SigmoidCross", new int[]{1}, input) {

            public Tenser<Tensor> compute() {
                Tensor a = getInput()[0], b = getInput()[1];
                return new Tenser<>(minus(add(mul(a, log(b)), mul(minus(cons(1), a), log(minus(cons(1), b))))));
            }

        };
    }

    public Tensor square(Tensor... input) {
        return new TensorFunction("Square", new int[]{1}, input) {

            public Tenser<Tensor> compute() {
                Tensor a = getInput()[0], b = getInput()[1];
                return new Tenser<>(mul(cons(0.5), pow(minus(a, b), cons(2d))));
            }

        };
    }

}