package com.deep.framework.core;

import com.deep.framework.cuda.Cublas;
import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.annotation.Cuda;

import java.io.Serializable;

import static com.deep.framework.lang.ForEach.forEach;
import static com.deep.framework.lang.Shape.*;
import static java.lang.Math.atan;

public class TensorFlow implements Serializable {

    public Tensor add(Tensor... input) {
        return new ScalarOperator("Add", input) {

            @Cuda
            public double compute() {
                return inputStream().mapToDouble(a -> a.getValue()).sum();
            }

            public void gradient(double grad) {
                inputStream().forEach(a -> a.setGrad(grad));
            }

        };
    }

    public Tensor addx(Tensor... input) {
        return new TensorOperator("Addx", input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> B = createOutput(getInput(0));
                inputStream().forEach(A -> {
                    forEach(A, B, (Tensor a, Tensor b) -> b.setValue(b.getValue() + a.getValue()));
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> B = getOutput();
                inputStream().forEach(A -> {
                    forEach(A, B, (Tensor a, Tensor b) -> a.setGrad(b.getGrad()));
                });
            }

        };
    }

    public Tensor minus(Tensor... input) {
        return new ScalarOperator("Minus", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return valx - valy;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                inx.setGrad(grad);
                iny.setGrad(-grad);
            }

        };
    }


    public Tensor minus(Tensor input) {
        return new ScalarOperator("Minusx", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return -valx;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                inx.setGrad(-grad);
            }

        };
    }

    public Tensor mul(Tensor... input) {
        return new ScalarOperator("Mul", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return valx * valy;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(grad * valy);
                iny.setGrad(grad * valx);
            }

        };
    }

    public Tensor div(Tensor... input) {
        return new ScalarOperator("Div", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return valx / valy;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(grad / valy);
                iny.setGrad(-grad * valx / Math.pow(valy, 2));
            }

        };
    }

    public Tensor exp(Tensor... input) {
        return new ScalarOperator("Exp", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.exp(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.exp(valx));
            }

        };
    }

    public Tensor expx(Tensor input) {
        return new TensorOperator("Expx", input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = zeroTensors(A);
                forEach(A, B, (Tensor a, Tensor b) -> b.setValue(Math.exp(a.getValue())));
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getOutput();
                forEach(A, B, (Tensor a, Tensor b) -> a.setGrad(b.getGrad() * b.getValue()));
            }

        };
    }

    public Tensor pow(Tensor... input) {
        return new ScalarOperator("Pow", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return Math.pow(valx, valy);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(grad * valy * Math.pow(valx, valy - 1));
            }

        };
    }

    public Tensor log(Tensor... input) {
        return new ScalarOperator("Log", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.log(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / valx);
            }

        };
    }

    public Tensor sum(Tensor input) {
        return new ScalarOperator("Sum", input) {

            public double compute() {
                Object A = getInput(0);
                Tensor B = new Tensor(0d);
                forEach(A, (Tensor a) -> B.setValue(B.getValue() + a.getValue()));
                return B.getValue();
            }

            public void gradient(double grad) {
                Object A = getInput(0);
                forEach(A, (Tensor a) -> a.setGrad(grad));
            }

        };
    }

    public Tensor sin(Tensor... input) {
        return new ScalarOperator("Sin", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.sin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.cos(valx));
            }

        };
    }

    public Tensor cos(Tensor... input) {
        return new ScalarOperator("Cos", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.cos(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * -Math.sin(valx));
            }

        };
    }

    public Tensor tan(Tensor... input) {
        return new ScalarOperator("Tan", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.tan(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.pow(1 / Math.cos(valx), 2));
            }

        };
    }

    public Tensor cot(Tensor... input) {
        return new ScalarOperator("Cot", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.cos(valx) / Math.sin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * -Math.pow(1 / Math.sin(valx), 2));
            }

        };
    }

    public Tensor sec(Tensor... input) {
        return new ScalarOperator("Sec", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return 1 / Math.cos(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.tan(valx) / Math.cos(valx));
            }

        };
    }

    public Tensor csc(Tensor... input) {
        return new ScalarOperator("Csc", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return 1 / Math.sin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * -Math.cos(valx) / Math.pow(Math.sin(valx), 2));
            }

        };
    }

    public Tensor arcsin(Tensor... input) {
        return new ScalarOperator("Arcsin", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.asin(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / Math.pow(1 - Math.pow(valx, 2), -2));
            }

        };
    }

    public Tensor arccos(Tensor... input) {
        return new ScalarOperator("Arccos", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.acos(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / -Math.pow(1 - Math.pow(valx, 2), -2));
            }

        };
    }

    public Tensor arctan(Tensor... input) {
        return new ScalarOperator("Arctan", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return Math.atan(valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / (1 + Math.pow(valx, 2)));
            }

        };
    }

    public Tensor arccot(Tensor... input) {
        return new ScalarOperator("Arccot", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return atan(1 / valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / -(1 + Math.pow(valx, 2)));
            }

        };
    }

    public Tensor relu(Tensor input) {
        return new ScalarOperator("Relu", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                return valx > 0 ? valx : 0.1 * valx;
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(valx > 0 ? grad : 0.1 * grad);
            }

        };
    }

    public Tensor relux(Tensor input) {
        return new TensorFunction("Relux", input) {

            @Cuda
            public Object compute() {
                Object A = getInput(0), B = zeroTensors(A);
                forEach(A, B, (Tensor a, Tenser<Tensor> b, int i) -> {
                    b.set(relu(a), i);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor max(Tensor... input) {
        return new ScalarOperator("Max", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return Math.max(valx, valy);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(valx > valy ? grad : 0);
                iny.setGrad(valx < valy ? grad : 0);
            }

        };
    }

    public Tensor funcx(Tenser input) {
        return new TensorFunction(input);
    }

    public Tensor cons(double value) {
        return new TensorConst(value);
    }

    public Tensor cons(int[] shape, double value) {
        return new TensorConst(shape, value);
    }

    public Tensor matmul(Tensor... input) {
        return new TensorOperator("Matmul", input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = createOutput(new int[]{A.shape(0), B.shape(1)});
                Cublas.New().matmul(getInput()[0], getInput()[1], this);
                return C;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = getOutput();
                Cublas.New().matmulGrad(getInput()[0], getInput()[1], this);
            }

        };

    }

    public Tensor matmulTran(Tensor... input) {
        return new TensorOperator("MatmulTran", input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = createOutput(new int[]{A.shape(0), B.shape(0)});
                forEach(A.shape(0), B.shape(0), A.shape(1), (i, l, j) -> {
                    Tensor inx = A.get(i, j), iny = B.get(l, j), out = C.get(i, l);
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = getOutput();
                forEach(A.shape(0), B.shape(0), A.shape(1), (i, l, j) -> {
                    Tensor inx = A.get(i, j), iny = B.get(l, j), out = C.get(i, l);
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setGrad(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor shape(Tensor... input) {
        return new TensorFunction("Shape", input) {

            public Object compute() {
                Object A = getInput(0), B = getInput(1);
                Object C = zeroTensors(B);
                reshape(A, C);
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor prod(Tensor... input) {
        return new TensorFunction("Prod", input) {

            public Object compute() {
                Object A = getInput(0), C = zeroTensors(A);
                Tensor b = getInput(1);
                forEach(A, C, (Tensor a, Tenser<Tensor> c, int i) -> {
                    c.set(mul(a, b), i);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoid(Tensor input) {
        return new ScalarFunction("Sigmoid", input) {

            public Tensor compute() {
                Tensor A = getInput(0);
                return div(cons(1d), add(cons(1d), exp(minus(A))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidx(Tensor input) {
        return new TensorFunction("Sigmoidx", input) {

            @Cuda
            public Object compute() {
                Object A = getInput(0), B = zeroTensors(A);
                forEach(A, B, (Tensor a, Tenser<Tensor> b, int i) -> {
                    b.set(sigmoid(a), i);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor square(Tensor... input) {
        return new ScalarFunction("Square", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return mul(cons(0.5), pow(minus(a, b), cons(2d)));
            }

            public void gradient() { }

        };
    }

    public Tensor squarex(Tensor... input) {
        return new ScalarFunction("Squarex", input) {

            @Cuda
            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
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
                Tensor a = getInput(0), b = getInput(1);
                return minus(mul(a, log(b)));
            }

            public void gradient() { }

        };
    }

    public Tensor softmaxCrossx(Tensor... input) {
        return new ScalarFunction("SoftmaxCrossx", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
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
                Tensor a = getInput(0), b = getInput(1);
                return minus(add(mul(a, log(b)), mul(minus(cons(1), a), log(minus(cons(1), b)))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidCrossx(Tensor... input) {
        return new ScalarFunction("SigmoidCrossx", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {cons(0d)};
                forEach(A, B, (Tensor a, Tensor b) -> {
                    C[0] = add(C[0], sigmoidCross(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor conv(int[] stride, int padding, Tensor... input) {
        return new TensorOperator("Conv", input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = padding(getInput(1), padding);
                int heights = stride[0], widths = stride[1];
                int height = (B.shape(0) - A.shape(0)) / heights + 1;
                int width = (B.shape(1) - A.shape(1)) / widths + 1;
                Tenser<Tensor> C = createOutput(new int[]{height, width});
                forEach(height, width, A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h * heights + m, w * widths + n), out = C.get(h, w);
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = padding(getInput(1), padding);
                Tenser<Tensor> C = getOutput();
                int heights = stride[0], widths = stride[1];
                forEach(C.shape(0), C.shape(1), A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h * heights + m, w * widths + n), out = C.get(h, w);
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setGrad(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor convx(int[] stride, int padding, Tensor... input) {
        return new TensorFunction("Convx", input) {

            public Object compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                int heighs = stride[0], widths = stride[1];
                int height = (B.shape(1) - A.shape(1) + 2 * padding) / heighs + 1;
                int width = (B.shape(2) - A.shape(2) + 2 * padding) / widths + 1;
                Tenser<Tensor> C = zeroTensors(new int[]{A.shape(0)}, new int[]{height, width});
                forEach(B.shape(0), A.shape(0), (i, l) -> {
                    C.set(addx(C.get(l), conv(stride, padding,  funcx(A.get(l)), funcx(B.get(i)))), l);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor deconv(int[] stride, int padding, Tensor... input) {
        return new TensorOperator("Deconv", input) {

            public Tenser<Tensor> compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                int heighs = stride[0], widths = stride[1];
                int height = (B.shape(0) - 1) * heighs + A.shape(0) - 2 * padding;
                int width = (B.shape(1) - 1) * widths + A.shape(1) - 2 * padding;
                Tenser<Tensor> C = createOutput(new int[]{height, width});
                forEach(B.shape(0), B.shape(1), A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h, w), out = C.get(h * heighs + m, w * widths + n);
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                Tenser<Tensor> C = getOutput();
                int heighs = stride[0], widths = stride[1];
                forEach(B.shape(0), B.shape(1), A.shape(0), A.shape(1), (h, w, m, n) -> {
                    Tensor inx = A.get(m, n), iny = B.get(h, w), out = C.get(h * heighs + m, w * widths + n);
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setGrad(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor deconvx(int[] stride, int padding, Tensor... input) {
        return new TensorFunction("Deconvx", input) {

            public Object compute() {
                Tenser<Tensor> A = getInput(0), B = getInput(1);
                int heighs = stride[0], widths = stride[1];
                int height = (B.shape(1) - 1) * heighs + A.shape(1) - 2 * padding;
                int width = (B.shape(2) - 1) * widths + A.shape(2) - 2 * padding;
                Tenser<Tensor> C = zeroTensors(new int[]{A.shape(0)}, new int[]{height, width});
                forEach(B.shape(0), A.shape(0), (i, l) -> {
                    C.set(addx(C.get(l), deconv(stride, padding,  funcx(A.get(l)),  funcx(B.get(i)))), l);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor maxpool(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorOperator("Maxpool", input) {

            public Tenser compute() {
                Tenser<Tensor> A = padding(getInput(0), padding);
                int heighs = stride[0], widths = stride[1];
                int height = (A.shape(0) - kernelSize) / heighs + 1, width = (A.shape(1) - kernelSize) / widths + 1;
                Tenser<Tensor> B = createOutput(new int[]{height, width});
                forEach(height, width, kernelSize, kernelSize, (y, x, m, n) -> {
                    Tensor inx = A.get(y * heighs + m, x * widths + n), out = B.get(y, x);
                    out.setValue(Math.max(out.getValue(), inx.getValue()));
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = padding(getInput(0), padding), B = getOutput();
                int heighs = stride[0], widths = stride[1];
                forEach(B.shape(0), B.shape(1), kernelSize, kernelSize, (y, x, m, n) -> {
                    Tensor inx = A.get(y * heighs + m, x * widths + n), out = B.get(y, x);
                    inx.setGrad(inx.getValue() == out.getValue() ? out.getGrad() : 0d);
                });
            }

        };
    }

    public Tensor maxpoolx(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorFunction("Maxpoolx", input) {

            public Object compute() {
                Tenser<Tensor> A = getInput(0);
                Tenser<Tensor> B = zeroTensors(new int[]{A.shape(0)});
                forEach(A.shape(0), i -> {
                    B.set(maxpool(kernelSize, stride, padding,  funcx(A.get(i))), i);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor demaxpool(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorOperator("Demaxpool", input) {

            public Tenser compute() {
                Tenser<Tensor> A = getInput(0);
                int heighs = stride[0], widths = stride[1];
                int height = (A.shape(0) - 1) * heighs + kernelSize - 2 * padding;
                int width = (A.shape(1) - 1) * widths + kernelSize - 2 * padding;
                Tenser<Tensor> B = createOutput(new int[]{height, width});
                forEach(A.shape(0), A.shape(1), kernelSize, kernelSize, (y, x, m, n) -> {
                    Tensor inx = A.get(y, x), out = B.get(y * heighs + m, x * widths + n);
                    out.setValue(out.getValue() + inx.getValue());
                });
                return B;
            }

            public void gradient() {
                Tenser<Tensor> A = getInput(0);
                int heighs = stride[0], widths = stride[1];
                Tenser<Tensor> B = getOutput();
                forEach(A.shape(0), A.shape(1), kernelSize, kernelSize, (y, x, m, n) -> {
                    Tensor inx = A.get(y, x), out = B.get(y * heighs + m, x * widths + n);
                    inx.setGrad(out.getGrad());
                });
            }

        };
    }

    public Tensor demaxpoolx(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorFunction("Demaxpoolx", input) {

            public Object compute() {
                Tenser<Tensor> A = getInput(0);
                Tenser<Tensor> B = zeroTensors(new int[]{A.shape(0)});
                forEach(A.shape(0), i -> {
                    B.set(demaxpool(kernelSize, stride, padding, funcx(A.get(i))), i);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor softmax(Tensor input) {
        return new TensorFunction("Softmax", input) {

            @Cuda
            public Object compute() {
                Tenser<Tensor> A = getInput(0), B = zeroTensors(A);
                Tensor sum = sum(expx(funcx(A)));
                forEach(A, B, (Tensor a, Tenser<Tensor> b, int i) -> {
                    b.set(div(exp(a), sum), i);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor selfAttention(Tensor... input) {
        return new TensorFunction("SelfAttention", input) {

            public Object compute() {
                Tenser<Tensor> A = getInput(0);
                Tenser<Tensor> B = getInput(1);
                Tensor C0 = matmul(funcx(A), funcx(B.get(0)));
                Tensor C1 = matmul(funcx(A), funcx(B.get(1)));
                Tensor C2 = matmul(funcx(A), funcx(B.get(2)));
                return matmul(softmax(prod(matmulTran(C0, C1), cons(8))), C2);
            }

            public void gradient() { }

        };
    }

    public Tensor batchNorm(Tensor... input) {
        return new TensorFunction("BatchNorm", input) {

            public Object compute() {
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