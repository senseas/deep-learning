package com.deep.framework.graph;

import com.deep.framework.lang.function.For2;
import com.deep.framework.lang.function.Func1;
import com.deep.framework.lang.function.Func2;

import java.io.Serializable;

import static com.deep.framework.lang.Shape.*;

public class TensorFlow implements Serializable {

    public Tensor add(Tensor... input) {
        return new TensorOparetor("Add", input) {

            public None compute() {
                double value = inputStream().mapToDouble(a -> ((None) a).getValue()).sum();
                return new None(value);
            }

            public void gradient() {
                None out = getOutput();
                inputStream().forEach(a -> ((None) a).setGrad(out.getGrad()));
            }

        };
    }

    public Tensor addx(Tensor... input) {
        return new TensorOparetor("Addx", input) {

            public Object compute() {
                Object B = zeroNones(getInput(0));
                inputStream().forEach(A -> {
                    farEach(A, B, (Func2<None>) (a, b) -> b.setValue(b.getValue() + a.getValue()));
                });
                return B;
            }

            public void gradient() {
                Object B = getOutput();
                inputStream().forEach(A -> {
                    farEach(A, B, (Func2<None>) (a, b) -> a.setGrad(b.getGrad()));
                });
            }

        };
    }

    public Tensor minus(Tensor... input) {
        return new TensorOparetor("Minus", input) {

            public None compute() {
                if (input.length == 1) {
                    None inx = getInput(0);
                    double valx = inx.getValue();
                    return new None(-valx);
                } else {
                    None inx = getInput(0), iny = getInput(1);
                    double valx = inx.getValue(), valy = iny.getValue();
                    return new None(valx - valy);
                }
            }

            public void gradient() {
                if (input.length == 1) {
                    None inx = getInput(0), out = getOutput();
                    double grad = out.getGrad();
                    inx.setGrad(-grad);
                } else {
                    None inx = getInput(0), iny = getInput(1), out = getOutput();
                    double grad = out.getGrad();
                    inx.setGrad(grad);
                    iny.setGrad(-grad);
                }
            }

        };
    }

    public Tensor mul(Tensor... input) {
        return new TensorOparetor("Mul", input) {

            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(valx * valy);
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad * valy);
                iny.setGrad(grad * valx);
            }

        };
    }

    public Tensor div(Tensor... input) {
        return new TensorOparetor("Div", input) {

            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(valx / valy);
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad / valy);
                iny.setGrad(-grad * valx / (valy * valy));
            }

        };
    }

    public Tensor exp(Tensor... input) {
        return new TensorOparetor("Exp", input) {

            public None compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return new None(Math.exp(valx));
            }

            public void gradient() {
                None inx = getInput(0), out = getOutput();
                double valx = inx.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad * Math.exp(valx));
            }

        };
    }

    public Tensor expx(Tensor input) {
        return new TensorFunction("Expx", input) {

            public Object compute() {
                Object A = getInput(0), B = zeroTensors(A);
                forEach(A, B, (For2<Tensor>) (a, b, i) -> {
                    b[i] = exp(a);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor pow(Tensor... input) {
        return new TensorOparetor("Pow", input) {

            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(Math.pow(valx, valy));
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad * valy * Math.pow(valx, valy - 1));
            }

        };
    }

    public Tensor log(Tensor... input) {
        return new TensorOparetor("Log", input) {

            public None compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return new None(Math.log(valx));
            }

            public void gradient() {
                None inx = getInput(0), out = getOutput();
                double valx = inx.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad / valx);
            }

        };
    }

    public Tensor relu(Tensor input) {
        return new TensorOparetor("Relu", input) {

            public None compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return new None(valx > 0 ? valx : 0.1 * valx);
            }

            public void gradient() {
                None inx = getInput(0), out = getOutput();
                double valx = inx.getValue();
                double grad = out.getGrad();
                inx.setGrad(valx > 0 ? grad : 0.1 * grad);
            }

        };
    }

    public Tensor relux(Tensor input) {
        return new TensorOparetor("Relux", input) {

            public Object compute() {
                Object A = getInput(0), B = zeroNones(A);
                farEach(A, B, (Func2<None>) (a, b) -> {
                    double value = a.getValue();
                    b.setValue(value > 0 ? value : 0.1 * value);
                });
                return B;
            }

            public void gradient() {
                Object A = getInput(0), B = getOutput();
                farEach(A, B, (Func2<None>) (a, b) -> {
                    double grad = b.getGrad();
                    a.setGrad(a.getValue() > 0 ? grad : 0.1 * grad);
                });
            }

        };
    }

    public Tensor max(Tensor... input) {
        return new TensorOparetor("Max", input) {

            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(Math.max(valx, valy));
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(valx > valy ? grad : 0);
                iny.setGrad(valx < valy ? grad : 0);
            }

        };
    }

    public Tensor matmul(Tensor... input) {
        return new TensorOparetor("Matmul", input) {

            public Object compute() {
                None[][] A = getInput(0), B = getInput(1);
                None[][] C = zeroNones(new None[A.length][B[0].length]);
                forEach(A.length, B[0].length, A[0].length, (i, l, j) -> {
                    None inx = A[i][j], iny = B[j][l], out = C[i][l];
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                None[][] A = getInput(0), B = getInput(1);
                None[][] C = getOutput();
                forEach(A.length, B[0].length, A[0].length, (i, l, j) -> {
                    None inx = A[i][j], iny = B[j][l], out = C[i][l];
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setGrad(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor matmulTran(Tensor... input) {
        return new TensorOparetor("MatmulTran", input) {

            public Object compute() {
                None[][] A = getInput(0), B = getInput(1);
                None[][] C = zeroNones(new None[A.length][B.length]);
                forEach(A.length, B.length, A[0].length, (i, l, j) -> {
                    None inx = A[i][j], iny = B[l][j], out = C[i][l];
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                None[][] A = getInput(0), B = getInput(1);
                None[][] C = getOutput();
                forEach(A.length, B.length, A[0].length, (i, l, j) -> {
                    None inx = A[i][j], iny = B[l][j], out = C[i][l];
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
                forEach(A, C, (For2<Tensor>) (a, c, i) -> {
                    c[i] = mul(a, b);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoid(Tensor input) {
        return new TensorFunction("Sigmoid", input) {

            public Tensor compute() {
                Tensor A = getInput(0);
                return div(new TensorConst(1d), add(new TensorConst(1d), exp(minus(A))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidx(Tensor input) {
        return new TensorFunction("Sigmoidx", input) {

            public Object compute() {
                Object A = getInput(0), B = zeroTensors(A);
                forEach(A, B, (For2<Tensor>) (a, b, i) -> {
                    b[i] = sigmoid(a);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor square(Tensor... input) {
        return new TensorFunction("Square", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return mul(new TensorConst(0.5), pow(minus(a, b), new TensorConst(2d)));
            }

            public void gradient() { }

        };
    }

    public Tensor squarex(Tensor... input) {
        return new TensorFunction("Squarex", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new TensorConst(0d)};
                forEach(A, B, (Func2<Tensor>) (a, b) -> {
                    C[0] = add(C[0], square(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor softmaxCross(Tensor... input) {
        return new TensorFunction("SoftmaxCross", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return minus(mul(a, log(b)));
            }

            public void gradient() { }

        };
    }

    public Tensor softmaxCrossx(Tensor... input) {
        return new TensorFunction("SoftmaxCrossx", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new TensorConst(0d)};
                forEach(A, B, (Func2<Tensor>) (a, b) -> {
                    C[0] = add(C[0], softmaxCross(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidCross(Tensor... input) {
        return new TensorFunction("SigmoidCross", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return minus(add(mul(a, log(b)), mul(minus(new TensorConst(1), a), log(minus(new TensorConst(1), b)))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidCrossx(Tensor... input) {
        return new TensorFunction("SigmoidCrossx", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new TensorConst(0d)};
                forEach(A, B, (Func2<Tensor>) (a, b) -> {
                    C[0] = add(C[0], sigmoidCross(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor sum(Tensor input) {
        return new TensorOparetor("Sum", input) {

            public None compute() {
                Object A = getInput(0);
                None B = new None(0d);
                farEach(A, (Func1<None>) a -> B.setValue(B.getValue() + a.getValue()));
                return B;
            }

            public void gradient() {
                Object A = getInput(0);
                None B = getOutput();
                farEach(A, (Func1<None>) a -> a.setGrad(B.getGrad()));
            }

        };
    }

    public Tensor conv(int[] stride, int padding, Tensor... input) {
        return new TensorOparetor("Conv", input) {

            public None[][] compute() {
                None[][] A = getInput(0), B = padding(getInput(1), padding);
                int heights = stride[0], widths = stride[1];
                int height = (B.length - A.length) / heights + 1, width = (B[0].length - A[0].length) / widths + 1;
                None[][] C = zeroNones(new None[height][width]);
                farEach(height, width, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h * heights + m][w * widths + n], out = C[h][w];
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                None[][] A = getInput(0), B = padding(getInput(1), padding);
                None[][] C = getOutput();
                int heights = stride[0], widths = stride[1];
                farEach(C.length, C[0].length, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h * heights + m][w * widths + n], out = C[h][w];
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setGrad(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor convx(int[] stride, int padding, Tensor... input) {
        return new TensorFunction("Convx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0), B = getInput(1);
                int heighs = stride[0], widths = stride[1];
                int height = (B[0].length - A[0].length + 2 * padding) / heighs + 1, width = (B[0][0].length - A[0][0].length + 2 * padding) / widths + 1;
                Tensor[] C = zeroTensors(new Tensor[A.length], new int[]{height, width});
                farEach(B.length, A.length, (i, l) -> {
                    C[l] = addx(C[l], conv(stride, padding, new Tensor(A[l]), new Tensor(B[i])));
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor deconv(int[] stride, int padding, Tensor... input) {
        return new TensorOparetor("Deconv", input) {

            public None[][] compute() {
                None[][] A = getInput(0), B = getInput(1);
                int heighs = stride[0], widths = stride[1];
                int height = (B.length - 1) * heighs + A.length - 2 * padding, width = (B[0].length - 1) * widths + A[0].length - 2 * padding;
                None[][] C = zeroNones(new None[height][width]);
                forEach(B.length, B[0].length, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h][w], out = C[h * heighs + m][w * widths + n];
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                None[][] A = getInput(0), B = getInput(1);
                None[][] C = getOutput();
                int heighs = stride[0], widths = stride[1];
                forEach(B.length, B[0].length, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h][w], out = C[h * heighs + m][w * widths + n];
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setGrad(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor deconvx(int[] stride, int padding, Tensor... input) {
        return new TensorFunction("Deconvx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0), B = getInput(1);
                int heighs = stride[0], widths = stride[1];
                int height = (B[0].length - 1) * heighs + A[0].length - 2 * padding, width = (B[0][0].length - 1) * widths + A[0][0].length - 2 * padding;
                Tensor[] C = zeroTensors(new Tensor[A.length], new int[]{height, width});
                forEach(B.length, A.length, (i, l) -> {
                    C[l] = addx(C[l], deconv(stride, padding, new Tensor(A[l]), new Tensor(B[i])));
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor maxpool(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorOparetor("Maxpool", input) {

            public None[][] compute() {
                None[][] A = padding(getInput(0), padding);
                int heighs = stride[0], widths = stride[1];
                int height = (A.length - kernelSize) / heighs + 1, width = (A[0].length - kernelSize) / widths + 1;
                None[][] B = zeroNones(new None[height][width]);
                forEach(height, width, kernelSize, kernelSize, (y, x, m, n) -> {
                    None inx = A[y * heighs + m][x * widths + n], out = B[y][x];
                    out.setValue(Math.max(out.getValue(), inx.getValue()));
                });
                return B;
            }

            public void gradient() {
                None[][] A = padding(getInput(0), padding), B = getOutput();
                int heighs = stride[0], widths = stride[1];
                forEach(B.length, B[0].length, kernelSize, kernelSize, (y, x, m, n) -> {
                    None inx = A[y * heighs + m][x * widths + n], out = B[y][x];
                    inx.setGrad(inx.getValue() == out.getValue() ? out.getGrad() : 0d);
                });
            }

        };
    }

    public Tensor maxpoolx(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorFunction("Maxpoolx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0);
                Tensor[] B = new Tensor[A.length];
                forEach(A.length, i -> {
                    B[i] = maxpool(kernelSize, stride, padding, new Tensor(A[i]));
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor demaxpool(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorOparetor("Demaxpool", input) {

            public None[][] compute() {
                None[][] A = getInput(0);
                int heighs = stride[0], widths = stride[1];
                int height = (A.length - 1) * heighs + kernelSize - 2 * padding, width = (A[0].length - 1) * widths + kernelSize - 2 * padding;
                None[][] B = zeroNones(new None[height][width]);
                forEach(A.length, A[0].length, kernelSize, kernelSize, (y, x, m, n) -> {
                    None inx = A[y][x], out = B[y * heighs + m][x * widths + n];
                    out.setValue(out.getValue() + inx.getValue());
                });
                return B;
            }

            public void gradient() {
                None[][] A = getInput(0);
                int heighs = stride[0], widths = stride[1];
                None[][] B = getOutput();
                forEach(A.length, A[0].length, kernelSize, kernelSize, (y, x, m, n) -> {
                    None inx = A[y][x], out = B[y * heighs + m][x * widths + n];
                    inx.setGrad(out.getGrad());
                });
            }

        };
    }

    public Tensor demaxpoolx(int kernelSize, int[] stride, int padding, Tensor input) {
        return new TensorFunction("Demaxpoolx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0);
                Tensor[] B = new Tensor[A.length];
                forEach(A.length, i -> {
                    B[i] = demaxpool(kernelSize, stride, padding, new Tensor(A[i]));
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor softmax(Tensor input) {
        return new TensorFunction("Softmax", input) {

            public Object compute() {
                Object[] A = getInput(0), B = zeroTensors(A);
                forEach(A, B, (For2<Tensor>) (a, b, i) -> {
                    b[i] = div(exp(a), sum(expx(new Tensor(A))));
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor selfAttention(Tensor... input) {
        return new TensorFunction("Softmax", input) {

            public Object compute() {
                Tensor[][] A = getInput(0);
                Tensor[][][] B = getInput(1);
                Tensor C0 = matmul(new Tensor(A), new Tensor(B[0]));
                Tensor C1 = matmul(new Tensor(A), new Tensor(B[1]));
                Tensor C2 = matmul(new Tensor(A), new Tensor(B[2]));
                return matmul(softmax(prod(matmulTran(C0, C1), new TensorConst(8))), C2);
            }

            public void gradient() { }

        };
    }

    public Tensor batchNorm(Tensor... input) {
        return new TensorFunction("BatchNorm", input) {

            public Object compute() {
                Object[] A = getInput(0), B = zeroTensors(A);
                Tensor C = mul(new TensorConst(1d / A.length), sum(new Tensor(A)));
                Tensor[] D = {new TensorConst(0)};
                forEach(A, a -> D[0] = add(D[0], pow(minus((Tensor) a, C), new TensorConst(2))));
                Tensor E = pow(add(mul(new TensorConst(1d / A.length), D[0]), new TensorConst(Math.E)), new TensorConst(0.5));
                forEach(A, B, (For2<Tensor>) (a, b, i) -> b[i] = add(mul(new Tensor(0.9), div(minus(a, C), E)), new Tensor(0.9)));
                return B;
            }

            public void gradient() { }

        };
    }

}