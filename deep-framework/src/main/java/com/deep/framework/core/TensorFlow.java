package com.deep.framework.core;

import com.deep.framework.cudnn.Reduce;
import com.deep.framework.graph.*;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenser;

import java.io.Serializable;
import java.util.Arrays;

import static com.deep.framework.cuda.Matmul.*;
import static com.deep.framework.cudnn.Softmax.softmaxBackward;
import static com.deep.framework.cudnn.Softmax.softmaxForward;
import static com.deep.framework.lang.ForEach.forEach;
import static com.deep.framework.lang.Shape.*;
import static jcuda.jcudnn.cudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_AVG;

public class TensorFlow implements Serializable {

    public Tensor funcx(Tenser input) {
        return new TensorFunction(input);
    }

    public Tensor cons(double value) {
        return new TensorConst(value);
    }

    public Tensor cons(double value, int[] shape) {
        return new TensorConst(value, shape);
    }

    public Tensor add(Tensor... input) {
        return new ScalarOperator("Add", input) {

            public double compute() {
                return Arrays.stream(getInput()).mapToDouble(Tensor::data).sum();
            }

            public void gradient(double grad) {
                Arrays.stream(getInput()).forEach(a -> a.grad(grad));
            }

        };
    }

    public Tensor addx(Tensor inx, Tensor iny) {
        return new TensorOperator("Addx", inx.getShape(), inx, iny) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> B = getOutput();
                Arrays.stream(getInput()).map(Tensor::getOutput).forEach(A -> {
                    forEach(A, B, (Tensor a, Tensor b) -> b.data(b.data() + a.data()));
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> B = getOutput();
                Arrays.stream(getInput()).map(Tensor::getOutput).forEach(A -> {
                    forEach(A, B, (Tensor a, Tensor b) -> a.grad(b.grad()));
                });
            }

        };
    }

    public Tensor minus(Tensor... input) {
        return new ScalarOperator("Minus", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                return valx - valy;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                inx.grad(grad);
                iny.grad(-grad);
            }

        };
    }


    public Tensor minus(Tensor input) {
        return new ScalarOperator("Minusx", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return -valx;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                inx.grad(-grad);
            }

        };
    }

    public Tensor mul(Tensor... input) {
        return new ScalarOperator("Mul", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                return valx * valy;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                inx.grad(grad * valy);
                iny.grad(grad * valx);
            }

        };
    }

    public Tensor div(Tensor... input) {
        return new ScalarOperator("Div", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                return valx / valy;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                inx.grad(grad / valy);
                iny.grad(-grad * valx / Math.pow(valy, 2));
            }

        };
    }

    public Tensor exp(Tensor... input) {
        return new ScalarOperator("Exp", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.exp(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad * Math.exp(valx));
            }

        };
    }

    public Tensor expx(Tensor input) {
        return new TensorOperator("Expx", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getOutput();
                forEach(A, B, (Tensor a, Tensor b) -> b.data(Math.exp(a.data())));
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getOutput();
                forEach(A, B, (Tensor a, Tensor b) -> a.grad(b.grad() * b.data()));
            }

        };
    }

    public Tensor pow(Tensor... input) {
        return new ScalarOperator("Pow", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                return Math.pow(valx, valy);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                inx.grad(grad * valy * Math.pow(valx, valy - 1));
            }

        };
    }

    public Tensor log(Tensor... input) {
        return new ScalarOperator("Log", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.log(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad / valx);
            }

        };
    }

    public Tensor sum(Tensor input) {
        return new ScalarOperator("Sum", input) {

            public double compute() {
                Tenser<Tensor> A = getInput(0).getOutput();
                return A.stream().mapToDouble(Tensor::data).sum();
            }

            public void gradient(double grad) {
                Tenser<Tensor> A = getInput(0).getOutput();
                A.stream().forEach(a -> a.grad(grad));
            }

        };
    }

    public Tensor sin(Tensor... input) {
        return new ScalarOperator("Sin", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.sin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad * Math.cos(valx));
            }

        };
    }

    public Tensor cos(Tensor... input) {
        return new ScalarOperator("Cos", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.cos(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad * -Math.sin(valx));
            }

        };
    }

    public Tensor tan(Tensor... input) {
        return new ScalarOperator("Tan", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.tan(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad * Math.pow(1 / Math.cos(valx), 2));
            }

        };
    }

    public Tensor cot(Tensor... input) {
        return new ScalarOperator("Cot", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.cos(valx) / Math.sin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad * -Math.pow(1 / Math.sin(valx), 2));
            }

        };
    }

    public Tensor sec(Tensor... input) {
        return new ScalarOperator("Sec", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return 1 / Math.cos(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad * Math.tan(valx) / Math.cos(valx));
            }

        };
    }

    public Tensor csc(Tensor... input) {
        return new ScalarOperator("Csc", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return 1 / Math.sin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad * -Math.cos(valx) / Math.pow(Math.sin(valx), 2));
            }

        };
    }

    public Tensor arcsin(Tensor... input) {
        return new ScalarOperator("Arcsin", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.asin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad / Math.pow(1 - Math.pow(valx, 2), -2));
            }

        };
    }

    public Tensor arccos(Tensor... input) {
        return new ScalarOperator("Arccos", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.acos(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad / -Math.pow(1 - Math.pow(valx, 2), -2));
            }

        };
    }

    public Tensor arctan(Tensor... input) {
        return new ScalarOperator("Arctan", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.atan(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad / (1 + Math.pow(valx, 2)));
            }

        };
    }

    public Tensor arccot(Tensor... input) {
        return new ScalarOperator("Arccot", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return Math.atan(1 / valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad / -(1 + Math.pow(valx, 2)));
            }

        };
    }

    public Tensor relu(Tensor input) {
        return new ScalarOperator("Relu", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return valx > 0 ? valx : 0.1 * valx;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(valx > 0 ? grad : 0.1 * grad);
            }

        };
    }

    public Tensor relux(Tensor input) {
        return new TensorOperator("Relux", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getOutput();
                forEach(A, B, (Tensor a, Tensor b) -> {
                    b.data(a.data() > 0 ? a.data() : 0.1 * a.data());
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getOutput();
                forEach(A, B, (Tensor a, Tensor b) -> {
                    a.grad(a.data() > 0 ? b.grad() : 0.1 * b.grad());
                });
            }

        };
    }

    public Tensor max(Tensor... input) {
        return new ScalarOperator("Max", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                return Math.max(valx, valy);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                inx.grad(valx > valy ? grad : 0);
                iny.grad(valx < valy ? grad : 0);
            }

        };
    }

    public Tensor matmul(Tensor... input) {
        return new TensorOperator("Matmul", Shape.shape(input[0].shape(0), input[1].shape(1)), input) {

            public Tenser<Tensor> compute() {
                matmulForward(getInput()[0], getInput()[1], this);
                return output;
            }

            public void gradient() {
                matmulBackward(getInput()[0], getInput()[1], this);
            }

        };

    }

    public Tensor matmulTran(Tensor... input) {
        return new TensorOperator("MatmulTran", Shape.shape(input[0].shape(0), input[1].shape(0)), input) {

            public Tenser<Tensor> compute() {
                matmulTranbForward(getInput()[0], getInput()[1], this);
                return output;
            }

            public void gradient() {
                matmulTranbBackward(getInput()[0], getInput()[1], this);
            }

        };
    }

    public Tensor matTran(Tensor input) {
        return new TensorOperator("MatmulTran", Shape.shape(input.shape(1), input.shape(0)), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0);
                Tenser<Tensor> B = getOutput();
                forEach(A.shape(0), A.shape(1), (i, l) -> {
                    Tensor inx = A.get(i, l), out = B.get(l, i);
                    out.data(inx.data());
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0);
                Tenser<Tensor> B = getOutput();
                forEach(A.shape(0), A.shape(1), (i, l) -> {
                    Tensor inx = A.get(i, l), out = B.get(l, i);
                    inx.grad(out.grad());
                });
            }

        };
    }

    public Tensor shape(Tensor... input) {
        return new TensorFunction("Shape", input[1].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = zeroTensors(B);
                reshape(A, C);
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor prod(Tensor... input) {
        return new TensorFunction("Prod", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), C = zeroTensors(A);
                Tensor b = getInput(1).one();
                forEach(A, C, (Tensor a) -> mul(a, b));
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoid(Tensor input) {
        return new ScalarFunction("Sigmoid", input) {

            public Tensor compute() {
                Tensor A = getInput(0).one();
                return div(cons(1d), add(cons(1d), exp(minus(A))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidx(Tensor input) {
        return new TensorFunction("Sigmoidx", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = zeroTensors(shape);
                forEach(A, B, (Tensor a) -> sigmoid(a));
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor square(Tensor... input) {
        return new ScalarFunction("Square", input) {

            public Tensor compute() {
                Tensor a = getInput(0).one(), b = getInput(1).one();
                return mul(cons(0.5), pow(minus(a, b), cons(2d)));
            }

            public void gradient() { }

        };
    }

    public Tensor squarex(Tensor... input) {
        return new ScalarFunction("Squarex", input) {

            public Tensor compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tensor[] C = {cons(0d)};
                forEach(A, B, (Tensor a, Tensor b) -> {
                    C[0] = add(C[0], square(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor softmaxCross(Tensor... input) {
        return new ScalarFunction("SoftmaxCross", input) {

            public Tensor compute() {
                Tensor a = getInput(0).one(), b = getInput(1).one();
                return minus(mul(a, log(b)));
            }

            public void gradient() {}

        };
    }

    public Tensor softmaxCrossx(Tensor... input) {
        return new ScalarFunction("SoftmaxCrossx", input) {

            public Tensor compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tensor[] C = {cons(0d)};
                forEach(A, B, (Tensor a, Tensor b) -> {
                    C[0] = add(C[0], softmaxCross(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidCross(Tensor... input) {
        return new ScalarFunction("SigmoidCross", input) {

            public Tensor compute() {
                Tensor a = getInput(0).one(), b = getInput(1).one();
                return minus(add(mul(a, log(b)), mul(minus(cons(1), a), log(minus(cons(1), b)))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidCrossx(Tensor... input) {
        return new ScalarFunction("SigmoidCrossx", input) {

            public Tensor compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tensor[] C = {cons(0d)};
                forEach(A, B, (Tensor a, Tensor b) -> {
                    C[0] = add(C[0], sigmoidCross(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor conv(int[] stride, int[] padding, Tensor... input) {
        Tensor A = input[0], B = input[1];
        int height = (B.shape(0) - A.shape(0) + 2 * padding[0]) / stride[0] + 1;
        int width = (B.shape(1) - A.shape(1) + 2 * padding[1]) / stride[1] + 1;

        return new TensorOperator("Conv", new int[]{height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = padding(getInput(1), padding);
                Tenser<Tensor> C = getOutput();
                forEach(height, width, A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h * stride[0] + m, w * stride[1] + n), out = C.get(h, w);
                    out.data(out.data() + inx.data() * iny.data());
                });
                return C;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = padding(getInput(1), padding);
                Tenser<Tensor> C = getOutput();
                forEach(C.shape(0), C.shape(1), A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h * stride[0] + m, w * stride[1] + n), out = C.get(h, w);
                    inx.grad(out.grad() * iny.data());
                    iny.grad(out.grad() * inx.data());
                });
            }

        };
    }

    public Tensor convx(int[] stride, int[] padding, Tensor... input) {
        Tensor A = input[0], B = input[1];
        int height = (B.shape(1) - A.shape(1) + 2 * padding[0]) / stride[0] + 1;
        int width = (B.shape(2) - A.shape(2) + 2 * padding[1]) / stride[1] + 1;

        return new TensorFunction("Convx", new int[]{A.shape(0), height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = zeroTensors(new int[]{A.shape(0)}, new int[]{height, width});
                forEach(B.shape(0), A.shape(0), (i, l) -> {
                    C.set(addx(C.get(l), conv(stride, padding, funcx(A.get(l)), funcx(B.get(i)))), l);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor deconv(int[] stride, int[] padding, Tensor... input) {
        Tensor A = input[0], B = input[1];
        int height = (B.shape(0) - 1) * stride[0] + A.shape(0) - 2 * padding[0];
        int width = (B.shape(1) - 1) * stride[1] + A.shape(1) - 2 * padding[1];

        return new TensorOperator("Deconv", new int[]{height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = getOutput();
                forEach(B.shape(0), B.shape(1), A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h, w), out = C.get(h * stride[0] + m, w * stride[1] + n);
                    out.data(out.data() + inx.data() * iny.data());
                });
                return C;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = getOutput();
                int heighs = stride[0], widths = stride[1];
                forEach(B.shape(0), B.shape(1), A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h, w), out = C.get(h * heighs + m, w * widths + n);
                    inx.grad(out.grad() * iny.data());
                    iny.grad(out.grad() * inx.data());
                });
            }

        };
    }

    public Tensor deconvx(int[] stride, int padding[], Tensor... input) {
        Tensor A = input[0], B = input[1];
        int height = (B.shape(1) - 1) * stride[0] + A.shape(1) - 2 * padding[0];
        int width = (B.shape(2) - 1) * stride[1] + A.shape(2) - 2 * padding[1];

        return new TensorFunction("Deconvx", new int[]{A.shape(0), height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = zeroTensors(new int[]{A.shape(0)}, new int[]{height, width});
                forEach(B.shape(0), A.shape(0), (i, l) -> {
                    C.set(addx(C.get(l), deconv(stride, padding, funcx(A.get(l)), funcx(B.get(i)))), l);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor maxpool(int[] window, int[] stride, int[] padding, Tensor input) {
        int height = (input.shape(0) - window[0] + 2 * padding[0]) / stride[0] + 1;
        int width = (input.shape(1) - window[1] + 2 * padding[1]) / stride[1] + 1;

        return new TensorOperator("Maxpool", new int[]{height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = padding(getInput(0), padding);
                Tenser<Tensor> B = getOutput();
                forEach(height, width, window[0], window[1], (y, x, m, n) -> {
                    Tensor inx = A.get(y * stride[0] + m, x * stride[1] + n), out = B.get(y, x);
                    out.data(Math.max(out.data(), inx.data()));
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = padding(getInput(0), padding), B = getOutput();
                forEach(B.shape(0), B.shape(1), window[0], window[1], (y, x, m, n) -> {
                    Tensor inx = A.get(y * stride[0] + m, x * stride[1] + n), out = B.get(y, x);
                    inx.grad(inx.data() == out.data() ? out.grad() : 0d);
                });
            }

        };
    }

    public Tensor maxpoolx(int[] window, int[] stride, int[] padding, Tensor input) {
        int height = (input.shape(1) - window[0] + 2 * padding[0]) / stride[0] + 1;
        int width = (input.shape(2) - window[1] + 2 * padding[1]) / stride[1] + 1;

        return new TensorFunction("Maxpoolx", new int[]{input.shape(0), height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0);
                Tenser<Tensor> B = zeroTensors(new int[]{A.shape(0)});
                forEach(A.shape(0), i -> {
                    B.set(maxpool(window, stride, padding, funcx(A.get(i))), i);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor demaxpool(int[] window, int[] stride, int[] padding, Tensor input) {
        int height = (input.shape(0) - 1) * stride[0] + window[0] - 2 * padding[0];
        int width = (input.shape(1) - 1) * stride[1] + window[0] - 2 * padding[1];

        return new TensorOperator("Demaxpool", new int[]{height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getOutput();
                forEach(A.shape(0), A.shape(1), window[0], window[1], (y, x, m, n) -> {
                    Tensor inx = A.get(y, x), out = B.get(y * stride[0] + m, x * stride[1] + n);
                    out.data(out.data() + inx.data());
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getOutput();
                forEach(A.shape(0), A.shape(1), window[0], window[1], (y, x, m, n) -> {
                    Tensor inx = A.get(y, x), out = B.get(y * stride[0] + m, x * stride[1] + n);
                    inx.grad(out.grad());
                });
            }

        };
    }

    public Tensor demaxpoolx(int[] window, int[] stride, int[] padding, Tensor input) {
        int height = (input.shape(1) - 1) * stride[0] + window[0] - 2 * padding[0];
        int width = (input.shape(2) - 1) * stride[1] + window[0] - 2 * padding[1];

        return new TensorFunction("Demaxpoolx", new int[]{input.shape(0), height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0);
                Tenser<Tensor> B = zeroTensors(new int[]{A.shape(0)});
                forEach(A.shape(0), i -> {
                    B.set(demaxpool(window, stride, padding, funcx(A.get(i))), i);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor softmax(Tensor input) {
        return new TensorOperator("Softmax", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                softmaxForward(getInput()[0], this);
                return output;
            }

            public void gradient() {
                softmaxBackward(getInput()[0], this);
            }

        };
    }

    public Tensor selfAttention(int scaler, Tensor... input) {
        return new TensorFunction("SelfAttention", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0];
                Tenser<Tensor> B = getInput(1);
                Tensor C0 = matmul(A, funcx(B.get(0)));
                Tensor C1 = matmul(A, funcx(B.get(1)));
                Tensor C2 = matmul(A, funcx(B.get(2)));
                Tensor C3 = matmulTran(C0, C1);
                Tensor C4 = softmax(mask(prod(C3, cons(scaler))));
                return new Tenser<>(matmul(C4, C2));
            }

            public void gradient() { }

        };
    }

    public Tensor multiHeadAttention(int scaler, Tensor... input) {
        return new TensorFunction("MultiHeadAttention", new int[]{input[0].shape(0), input[0].shape(1)}, input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0], C = getInput()[2], M = getInput()[3], N = getInput()[4];
                Tenser<Tensor> B = getInput(1);
                Tensor[] arr = new Tensor[B.shape(0)];
                forEach(arr.length, i -> arr[i] = selfAttention(scaler, A, funcx(B.get(i))));
                Tensor addx = addx(A, matmul(concat(arr), C));
                return new Tenser<>(layerNormal(addx, M, N));
            }

            public void gradient() {}

        };
    }

    public Tensor layerNormal(Tensor... input) {
        return new TensorFunction("LayerNormal", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1), C = getInput(2);
                Tensor mean = mean(funcx(A)), std = standard(funcx(A), mean);
                Tenser<Tensor> D = zeroTensors(A.shape);
                forEach(A, B, C, D, (Tensor a, Tensor b, Tensor c) -> {
                    return add(div(mul(b, minus(a, mean)), pow(add(std, cons(0.0000001)), cons(0.5))), c);
                });
                return D;
            }

            public void gradient() {}

        };
    }

    public Tensor standard(Tensor... input) {
        return new ScalarFunction("Standard", input) {

            public Tensor compute() {
                Tenser<Tensor> inx = getInput(0), iny = getInput(1);
                Tensor mean = iny.one();
                Tenser<Tensor> pows = new Tenser<>(Tensor.class, inx.shape);
                forEach(inx, pows, (Tensor a) -> pow(minus(a, mean), cons(2)));
                return mean(funcx(pows));
            }

            public void gradient() {}

        };
    }


    public Tensor mean(Tensor... input) {
        return new ScalarOperator("Mean", input) {

            public double compute() {
                Tensor inx = getInput(0);
                int[] inShape = Shape.shapes(inx.getShape()) , outShape = {1, 1, 1, 1};
                Reduce.reduce(inx.getData(), inShape, data, outShape, CUDNN_REDUCE_TENSOR_AVG);
                return data();
            }

            public void gradient() {}

        };
    }

    public Tensor wordEmbedding(Tensor... input) {
        return new ScalarFunction("WordEmbedding", input) {

            public Tensor compute() {
                Tensor A = getInput()[0], B = getInput()[1];
                return matmul(A, B);
            }

            public void gradient() {}

        };
    }

    public Tensor mask(Tensor... input) {
        return new TensorOperator("Mask", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor input = getInput()[0];
                for (int l = 0; l < shape(1); l++) {
                    for (int i = l; i < shape(0); i++) {
                        int idx = shape[1] * i + l;
                        data[idx] = input.getData()[idx];
                    }
                }
                return output;
            }

            public void gradient() {
                Tensor input = getInput()[0];
                for (int l = 0; l < shape(1); l++) {
                    for (int i = l; i < shape(0); i++) {
                        int idx = shape[1] * i + l;
                        input.getGrad()[idx] = grad[idx];
                    }
                }
            }

        };
    }

    public Tensor positionalEmbedding(int[] shape, Tensor... input) {
        return new TensorOperator("Mean", shape, input) {

            public Tenser compute() {
                Tensor inx = getInput()[0];
                int dim = shape[1];
                forEach(shape(0), shape(1), (int i, int l) -> {
                    if (l % 2 == 0) {
                        data[dim * i + l] = Math.sin(inx.getData()[i] / Math.pow(1000, 2 * l / dim));
                    } else {
                        data[dim * i + l] = Math.cos(inx.getData()[i] / Math.pow(1000, 2 * l / dim));
                    }
                });
                return output;
            }

            public void gradient() {}

        };
    }

    public Tensor concat(Tensor... input) {
        int M = input[0].shape(0), N = input[0].shape(1);

        return new TensorOperator("Concat", new int[]{M, N * input.length}, input) {

            public Tenser<Tensor> compute() {
                Tensor inx = getInput()[0];
                int M = inx.shape(0), N = inx.shape(1);

                for (int i = 0; i < input.length; i++) {
                    Tensor in = getInput()[i];
                    for (int m = 0; m < M; m++) {
                        for (int n = 0; n < N; n++) {
                            int x = m * N;
                            int idx = i * N + input.length * x + n;
                            int idy = x + n;
                            data[idx] = in.getData()[idy];
                        }
                    }
                }
                return output;
            }

            public void gradient() {
                for (int i = 0; i < input.length; i++) {
                    Tensor in = getInput()[i];
                    int M = in.shape(0), N = in.shape(1);
                    for (int m = 0; m < M; m++) {
                        for (int n = 0; n < N; n++) {
                            int x = m * N;
                            int idx = i * N + input.length * x + n;
                            int idy = x + n;
                            in.getGrad()[idy] += grad[idx];
                        }
                    }
                }
            }

        };
    }

    public Tensor batchNorm(Tensor... input) {
        return new TensorFunction("BatchNorm", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser A = getInput(0), B = zeroTensors(A);
                Tensor C = mul(cons(1d / A.shape(0)), sum(funcx(A)));
                Tensor[] D = {cons(0)};
                forEach(A, a -> D[0] = add(D[0], pow(minus((Tensor) a, C), cons(2))));
                Tensor E = pow(add(mul(cons(1d / A.shape(0)), D[0]), cons(Math.E)), cons(0.5));
                forEach(A, B, (Tensor a, Tenser<Tensor> b, int i) -> b.set(add(mul(new Tensor(0.9), div(minus(a, C), E)), new Tensor(0.9)), i));
                return B;
            }

            public void gradient() { }

        };
    }

}