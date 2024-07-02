package com.deep.framework.core;

import com.deep.framework.cudnn.Reduce;
import com.deep.framework.graph.*;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenser;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

import static com.deep.framework.cublas.Matmul.*;
import static com.deep.framework.cudnn.Activation.*;
import static com.deep.framework.cudnn.BatchNormal.normalBackward;
import static com.deep.framework.cudnn.BatchNormal.normalForward;
import static com.deep.framework.cudnn.OpTensor.*;
import static com.deep.framework.cudnn.Reduce.sumBackward;
import static com.deep.framework.cudnn.Softmax.softmaxBackward;
import static com.deep.framework.cudnn.Softmax.softmaxForward;
import static com.deep.framework.lang.ForEach.forEach;
import static com.deep.framework.lang.Shape.*;

public class TensorFlow implements Serializable {

    public Tensor Tensor(Tenser input) {
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
        int[] maxShape = shapeAligned(inx, iny);
        return new TensorOperator("Addx", maxShape, inx, iny) {

            public Tenser<Tensor> compute() {
                Arrays.stream(getInput()).forEach(A -> {
                    forEach(maxShape[0], i -> addTensorForward(A.get(i), this.get(i)));
                });
                return output;
            }

            public void gradient() {
                Arrays.stream(getInput()).forEach(A -> {
                    forEach(maxShape[0], i -> addTensorBackward(A.get(i), this.get(i)));
                });
            }

        };
    }

    public Tensor minus(Tensor... input) {
        int[] shaped = shapeAligned(input);
        return new TensorOperator("Minus", shaped, input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> {
                    double valx = a.data(), valy = b.data();
                    o.data(valx - valy);
                });
                return output;
            }

            public void gradient() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> {
                    a.grad(o.grad());
                    b.grad(-o.grad());
                });
            }

        };
    }

    public Tensor minus(Tensor input) {
        return new TensorOperator("Minusx", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(-a.data()));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(-o.grad()));
            }

        };
    }

    public Tensor mul(Tensor... input) {
        int[] shaped = shapeAligned(input);
        return new TensorOperator("Mul", shaped, input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> {
                    double valx = a.data(), valy = b.data();
                    o.data(valx * valy);
                });
                return output;
            }

            public void gradient() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> {
                    double valx = a.data(), valy = b.data();
                    a.grad(o.grad() * valy);
                    b.grad(o.grad() * valx);
                });
            }

        };
    }

    public Tensor div(Tensor... input) {
        return new TensorOperator("Div", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> {
                    double valx = a.data(), valy = b.data();
                    o.data(valx / valy);
                });
                return output;
            }

            public void gradient() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> {
                    double valx = a.data(), valy = b.data();
                    a.grad(o.grad() / valy);
                    b.grad(-o.grad() * valx / Math.pow(valy, 2));
                });
            }

        };
    }

    public Tensor exp(Tensor input) {
        return new TensorOperator("Exp", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.exp(a.data())));
                return O;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() * o.data()));
            }

        };
    }

    public Tensor pow(Tensor... input) {
        return new TensorOperator("Pow", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> o.data(Math.pow(a.data(), b.data())));
                return output;
            }

            public void gradient() {
                Tensor A = getInput()[0], B = getInput()[1], O = this;
                forEach(A, B, O, (Tensor a, Tensor b, Tensor o) -> a.grad(o.grad() * b.data() * Math.pow(a.data(), b.data() - 1)));
            }

        };
    }

    public Tensor log(Tensor input) {
        return new TensorOperator("Log", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.log(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() / a.data()));
            }

        };
    }

    public Tensor sum(Tensor input) {
        return new TensorOperator("Sum", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0];
                Reduce.sum(A, this, 0);
                return output;
            }

            public void gradient() {
                Tensor A = getInput()[0];
                sumBackward(A, this);
            }

        };
    }

    public Tensor sin(Tensor input) {
        return new TensorOperator("Sin", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.sin(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() * Math.cos(a.data())));
            }

        };
    }

    public Tensor cos(Tensor input) {
        return new TensorOperator("Cos", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.cos(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() * -Math.sin(a.data())));
            }

        };
    }

    public Tensor tan(Tensor input) {
        return new TensorOperator("Tan", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.tan(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() * Math.pow(1 / Math.cos(a.data()), 2)));
            }

        };
    }

    public Tensor cot(Tensor input) {
        return new TensorOperator("Cot", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.cos(a.data()) / Math.sin(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() * -Math.pow(1 / Math.sin(a.data()), 2)));
            }

        };
    }

    public Tensor sec(Tensor input) {
        return new TensorOperator("Sec", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(1 / Math.cos(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() * Math.tan(a.data()) / Math.cos(a.data())));
            }

        };
    }

    public Tensor csc(Tensor input) {
        return new TensorOperator("Csc", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(1 / Math.sin(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() * -Math.cos(a.data()) / Math.pow(Math.sin(a.data()), 2)));
            }

        };
    }

    public Tensor arcsin(Tensor input) {
        return new TensorOperator("Arcsin", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.asin(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() / Math.pow(1 - Math.pow(a.data(), 2), -2)));
            }

        };
    }

    public Tensor arccos(Tensor input) {
        return new TensorOperator("Arccos", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.acos(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() / -Math.pow(1 - Math.pow(a.data(), 2), -2)));
            }

        };
    }

    public Tensor arctan(Tensor input) {
        return new TensorOperator("Arctan", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.atan(a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() / (1 + Math.pow(a.data(), 2))));
            }

        };
    }

    public Tensor arccot(Tensor input) {
        return new TensorOperator("Arccot", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> o.data(Math.atan(1 / a.data())));
                return output;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), O = getOutput();
                forEach(A, O, (Tensor a, Tensor o) -> a.grad(o.grad() / -(1 + Math.pow(a.data(), 2))));
            }

        };
    }

    public Tensor relu(Tensor input) {
        return new TensorOperator("Relu", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                reluForward(getInput()[0], this);
                return output;
            }

            public void gradient() {
                reluBackward(getInput()[0], this);
            }

        };
    }

    public Tensor elu(Tensor input) {
        return new TensorOperator("Elu", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                eluForward(getInput()[0], this);
                return output;
            }

            public void gradient() {
                eluBackward(getInput()[0], this);
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
        int[] maxShape = shapeAligned(input);
        return new TensorOperator("Matmul", Shape.shape(maxShape[0], input[0].shape(1), input[1].shape(2)), input) {

            public Tenser<Tensor> compute() {
                forEach(maxShape[0], i -> matmulForward(getInput()[0].get(i), getInput()[1].get(i), this.get(i)));
                return output;
            }

            public void gradient() {
                forEach(maxShape[0], i -> matmulBackward(getInput()[0].get(i), getInput()[1].get(i), this.get(i)));
            }

        };

    }

    public Tensor matmulTran(Tensor inx, Tensor iny, Tensor alpha) {
        int[] shaped = shapeAligned(inx, iny);
        return new TensorOperator("MatmulTran", Shape.shape(shaped[0], inx.shape(1), iny.shape(1)), inx, iny) {

            public Tenser<Tensor> compute() {
                forEach(shaped[0], i -> matmulTranbForward(getInput()[0].get(i), getInput()[1].get(i), this.get(i), alpha));
                return output;
            }

            public void gradient() {
                forEach(shaped[0], i -> matmulTranbBackward(getInput()[0].get(i), getInput()[1].get(i), this.get(i), alpha));
            }

        };
    }

    public Tensor matTran(Tensor input) {
        return new TensorOperator("MatTran", Shape.shape(input.shape(1), input.shape(0)), input) {

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

        };
    }

    public Tensor prod(Tensor... input) {
        return new TensorOperator("Prod", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                mulTensorScalarForward(getInput()[0], getInput()[1], this);
                return output;
            }

            public void gradient() {
                mulTensorScalarBackward(getInput()[0], getInput()[1], this);
            }

        };
    }

    public Tensor sigmoid(Tensor input) {
        return new TensorOperator("Sigmoid", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0];
                sigmoidForward(A, this);
                return output;
            }

            public void gradient() {
                Tensor A = getInput()[0];
                sigmoidBackward(A, this);
            }

        };
    }

    public Tensor square(Tensor... input) {
        return new ScalarFunction("Square", input) {

            public Tensor compute() {
                Tensor a = getInput()[0], b = getInput()[1];
                return mul(cons(0.5), pow(minus(a, b), cons(2d)));
            }

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

        };
    }

    public Tensor softmaxCross(Tensor... input) {
        return new ScalarFunction("SoftmaxCross", input) {

            public Tensor compute() {
                Tensor a = getInput()[0], b = getInput()[1];
                return minus(mul(a, log(b)));
            }

        };
    }

    public Tensor softmaxCrossx(Tensor... input) {
        return new ScalarOperator("SoftmaxCrossx", input) {

            public double compute() {
                Tenser<Tensor> A = getInput(0).getOutput();
                Tenser<Tensor> B = getInput(1).getOutput();
                double[] C = {0d};
                forEach(A, B, (Tensor a, Tensor b) -> {
                    C[0] -= a.data() * Math.log(b.data());
                });
                return C[0];
            }

            public void gradient(double grad) {
                Tenser<Tensor> A = getInput(0).getOutput();
                Tenser<Tensor> B = getInput(1).getOutput();
                forEach(A, B, (Tensor a, Tensor b) -> {
                    a.grad(-grad * Math.log(b.data()));
                    b.grad(-grad * a.data() / b.data());
                });
            }

        };
    }

    public Tensor sigmoidCross(Tensor... input) {
        return new ScalarFunction("SigmoidCross", input) {

            public Tensor compute() {
                Tensor a = getInput()[0], b = getInput()[1];
                return minus(add(mul(a, log(b)), mul(minus(cons(1), a), log(minus(cons(1), b)))));
            }

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
                    C.set(addx(C.get(l), conv(stride, padding, Tensor(A.get(l)), Tensor(B.get(i)))), l);
                });
                return C;
            }

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

    public Tensor deconvx(int[] stride, int[] padding, Tensor... input) {
        Tensor A = input[0], B = input[1];
        int height = (B.shape(1) - 1) * stride[0] + A.shape(1) - 2 * padding[0];
        int width = (B.shape(2) - 1) * stride[1] + A.shape(2) - 2 * padding[1];

        return new TensorFunction("Deconvx", new int[]{A.shape(0), height, width}, input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = zeroTensors(new int[]{A.shape(0)}, new int[]{height, width});
                forEach(B.shape(0), A.shape(0), (i, l) -> {
                    C.set(addx(C.get(l), deconv(stride, padding, Tensor(A.get(l)), Tensor(B.get(i)))), l);
                });
                return C;
            }

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
                    B.set(maxpool(window, stride, padding, Tensor(A.get(i))), i);
                });
                return B;
            }

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
                    B.set(demaxpool(window, stride, padding, Tensor(A.get(i))), i);
                });
                return B;
            }

        };
    }

    public Tensor softmax(Tensor input, int axis) {
        return new TensorOperator("Softmax", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                softmaxForward(getInput()[0], this, axis);
                return output;
            }

            public void gradient() {
                softmaxBackward(getInput()[0], this, axis);
            }

        };
    }

    public Tensor expandx(Tensor input, int... shape) {
        return new TensorOperator("Expandx", shape, input) {

            public Tenser<Tensor> compute() {
                Tensor inx = getInput()[0];
                int a = size() / shape[0];
                for (int i = 0; i < shape[0]; i++) {
                    for (int l = 0; l < a; l++) {
                        data[i * a + l] = inx.getData()[l];
                    }
                }
                return output;
            }

            public void gradient() {
                Tensor inx = getInput()[0];
                int a = size() / shape[0];
                for (int i = 0; i < shape[0]; i++) {
                    for (int l = 0; l < a; l++) {
                        inx.getGrad()[l] += grad[i * a + l];
                    }
                }
            }
        };
    }

    public Tensor expand(Tensor input, int... shape) {
        return new TensorOperator("Expand", shape, input) {

            final Tensor inx = getInput()[0];
            final int a = getNext()[0], b = inx.getNext()[0], c = a / b;

            public Tenser<Tensor> compute() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int m = 0; m < c; m++) {
                        int o = m * b;
                        for (int n = 0; n < b; n++) {
                            data[d + o + n] = inx.getData()[e + n];
                        }
                    }
                }
                return output;
            }

            public void gradient() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int m = 0; m < c; m++) {
                        int o = m * b;
                        for (int n = 0; n < b; n++) {
                            inx.getGrad()[e + n] += grad[d + o + n];
                        }
                    }
                }
            }
        };
    }

    public Tensor expand(Tensor input, int[] in_shape, int[] shape) {
        return new TensorOperator("Expand", shape, input) {

            final Tensor inx = getInput()[0];
            final int a = getNext()[0], b = getNext(in_shape)[0], c = a / b;

            public Tenser<Tensor> compute() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int m = 0; m < c; m++) {
                        int o = m * b;
                        for (int n = 0; n < b; n++) {
                            data[d + o + n] = inx.getData()[e + n];
                        }
                    }
                }
                return output;
            }

            public void gradient() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int m = 0; m < c; m++) {
                        int o = m * b;
                        for (int n = 0; n < b; n++) {
                            inx.getGrad()[e + n] += grad[d + o + n];
                        }
                    }
                }
            }
        };
    }

    public Tensor expandre(Tensor input, int[] shape) {
        return new TensorOperator("expandre", shape, input) {

            final Tensor inx = getInput()[0];
            final int a = getNext()[0], b = inx.getNext()[0], c = a / b;

            public Tenser<Tensor> compute() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int n = 0; n < b; n++) {
                        int o = n * c;
                        for (int m = 0; m < c; m++) {
                            data[d + o + m] = inx.getData()[e + n];
                        }
                    }
                }
                return output;
            }

            public void gradient() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int n = 0; n < b; n++) {
                        int o = n * c;
                        for (int m = 0; m < c; m++) {
                            inx.getGrad()[e + n] += grad[d + o + m];
                        }
                    }
                }
            }
        };
    }

    public Tensor expandre(Tensor input, int[] in_shape, int[] shape) {
        return new TensorOperator("expandre", shape, input) {

            final  Tensor inx = getInput()[0];
            final int a = getNext()[0], b = getNext(in_shape)[0], c = a / b;

            public Tenser<Tensor> compute() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int n = 0; n < b; n++) {
                        int o = n * c;
                        for (int m = 0; m < c; m++) {
                            data[d + o + m] = inx.getData()[e + n];
                        }
                    }
                }
                return output;
            }

            public void gradient() {
                for (int l = 0; l < shape[0]; l++) {
                    int d = l * a, e = l * b;
                    for (int n = 0; n < b; n++) {
                        int o = n * c;
                        for (int m = 0; m < c; m++) {
                            inx.getGrad()[e + n] += grad[d + o + m];
                        }
                    }
                }
            }
        };
    }

    public Tensor layerNormal(Tensor input) {
        return new TensorFunction("LayerNormal", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0], B = new Tensor(new int[]{A.shape(1), A.shape(2)}), C = new Tensor(new int[]{A.shape(1), A.shape(2)});
                Tensor minus = minus(A, expand(mean(A, 1), shape));
                Tensor std = mean(pow(minus, cons(2, shape)), 1);
                Tensor a = pow(addx(std, cons(1.0E-7, std.getShape())), cons(0.5, std.getShape()));
                Tensor add = addx(mul(B, div(minus, expandre(a, shape))), C);
                return new Tenser<>(add);
            }

        };
    }

    public Tensor standard(Tensor... input) {
        return new ScalarFunction("Standard", input) {

            public Tensor compute() {
                Tenser<Tensor> inx = getInput(0);
                Tensor mean = getInput()[1], cons = cons(2);
                Tenser<Tensor> pows = zeroTensors(inx.shape);
                forEach(inx, pows, (Tensor a) -> pow(minus(a, mean), cons));
                return mean(Tensor(pows), 0);
            }

        };
    }

    public Tensor mean(Tensor input, int axis) {
        return new TensorOperator("Mean", new int[]{input.shape(0), input.shape(1)}, input) {

            public Tenser<Tensor> compute() {
                Tensor inx = getInput()[0];
                Reduce.mean(inx, this, axis);
                return output;
            }

            public void gradient() {
                Tensor inx = getInput()[0];
                int sizex = inx.getSize() / getSize();
                forEach(inx.getGrad().length, (int i) -> inx.getGrad()[i] += getGrad()[i / sizex] / sizex);
            }

        };
    }

    public Tensor mask(Tensor input) {
        return new TensorOperator("Mask", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor input = getInput()[0];
                int L = shape(0), N = shape(1), M = shape(2);
                for (int l = 0; l < L; l++) {
                    for (int m = 0; m < M; m++) {
                        for (int n = m; n < N; n++) {
                            int idx = l * M * N + M * n + m;
                            data[idx] = input.getData()[idx];
                        }
                    }
                }
                return output;
            }

            public void gradient() {
                Tensor input = getInput()[0];
                int L = shape(0), N = shape(1), M = shape(2);
                for (int l = 0; l < L; l++) {
                    for (int m = 0; m < M; m++) {
                        for (int n = m; n < N; n++) {
                            int idx = l * M * N + M * n + m;
                            input.getGrad()[idx] = grad[idx];
                        }
                    }
                }
            }

        };
    }

    public Tensor positionalEmbedding(int[] shape, Tensor... input) {
        return new TensorOperator("PositionalEmbedding", shape, input) {

            public Tenser<Tensor> compute() {
                Tensor input = getInput()[0];
                int L = shape[0], M = shape[1], N = shape[2];
                forEach(L, M, N, (int l, int m, int n) -> {
                    int index0 = l * M * N + m * N + n, index1 = l * M + m;
                    if (n % 2 == 0) {
                        data[index0] = Math.sin(input.getData()[index1] / Math.pow(1000, 2 * n / N));
                    } else {
                        data[index0] = Math.cos(input.getData()[index1] / Math.pow(1000, 2 * n / N));
                    }
                });
                return output;
            }

        };
    }

    public Tensor concat(Tensor... input) {
        int O = input[0].shape(0), M = input[0].shape(1), N = input[0].shape(2);

        return new TensorOperator("Concat", new int[]{O, M, N * input.length}, input) {

            public Tenser<Tensor> compute() {
                Tensor inx = getInput()[0];
                int I = input.length, L = inx.shape(0), M = inx.shape(1), N = inx.shape(2);
                for (int i = 0; i < I; i++) {
                    Tensor in = getInput()[i];
                    for (int l = 0; l < L; l++) {
                        for (int m = 0; m < M; m++) {
                            for (int n = 0; n < N; n++) {
                                int x = l * M * N + m * N;
                                int idx = i * N + I * x + n;
                                int idy = x + n;
                                data[idx] = in.getData()[idy];
                            }
                        }
                    }
                }
                return output;
            }

            public void gradient() {
                Tensor inx = getInput()[0];
                int I = input.length, L = inx.shape(0), M = inx.shape(1), N = inx.shape(2);
                for (int i = 0; i < I; i++) {
                    Tensor in = getInput()[i];
                    for (int l = 0; l < L; l++) {
                        for (int m = 0; m < M; m++) {
                            for (int n = 0; n < N; n++) {
                                int x = l * M * N + m * N;
                                int idx = i * N + I * x + n;
                                int idy = x + n;
                                in.getGrad()[idy] += grad[idx];
                            }
                        }
                    }
                }
            }

        };
    }

    public Tensor batchNormal(Tensor... input) {
        return new TensorOperator("BatchNormal", input[0].getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0], B = getInput()[1], C = getInput()[2];
                normalForward(A, B, C, this);
                return output;
            }

            public void gradient() {
                Tensor A = getInput()[0], B = getInput()[1], C = getInput()[2];
                normalBackward(A, B, C, this);
            }

        };
    }

    public Tensor linear(Tensor inx, Tensor iny) {
        int[] shaped = shapeAligned(inx, iny);
        int[] shape = {shaped[0], inx.shape(1), iny.shape(2)};
        return new TensorFunction("Linear", shape, inx, iny) {

            public Tenser<Tensor> compute() {
                Tensor tensor1 = matmul(getInput()[0], getInput()[1]);
                Tensor tensor2 = addx(tensor1, new Tensor(tensor1.getShape()));
                Tensor tensor3 = elu(tensor2);
                return new Tenser<>(tensor3);
            }

        };
    }

    public Tensor selfAttention(int dim, double scaler, Tensor input) {
        return new TensorFunction("SelfAttention", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0];
                Tensor C0 = matmul(A, new Tensor(new int[]{dim, dim}));
                Tensor C1 = matmul(A, new Tensor(new int[]{dim, dim}));
                Tensor C2 = matmul(A, new Tensor(new int[]{dim, dim}));
                Tensor C3 = matmulTran(C0, C1, cons(scaler));
                Tensor C4 = softmax(mask(C3), 1);
                return new Tenser<>(matmul(C4, C2));
            }

        };
    }

    public Tensor multiHeadAttention(int dim, int header_num, double scaler, Tensor input) {
        return new TensorFunction("MultiHeadAttention", input.getShape(), input) {

            public Tenser<Tensor> compute() {
                Tensor A = getInput()[0];
                Tensor[] attentions = IntStream.range(0, header_num).mapToObj(i -> selfAttention(dim, scaler, A)).toArray(Tensor[]::new);
                Tensor addx = addx(A, matmul(concat(attentions), new Tensor(new int[]{header_num * dim, dim})));
                Tensor normal = layerNormal(addx);
                return new Tenser<>(normal);
            }

        };
    }

    public Tensor transformer(int dim, int header_num, double scaler, Tensor... input) {
        return new TensorFunction("Transformer", new int[]{input[0].shape(0), input[0].shape(1)}, input) {

            public Tenser<Tensor> compute() {
                Tensor tensor11 = multiHeadAttention(dim, header_num, scaler, getInput()[0]);
                Tensor tensor12 = linear(tensor11, new Tensor(new int[]{dim, dim}));
                Tensor tensor13 = addx(tensor11, tensor12);
                Tensor tensor14 = layerNormal(tensor13);
                return new Tenser<>(tensor14);
            }

        };
    }

}