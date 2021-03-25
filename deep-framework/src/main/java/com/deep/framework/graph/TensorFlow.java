package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.annotation.Operator;
import com.deep.framework.lang.function.Func2;

import java.util.stream.IntStream;

public class TensorFlow extends Shape {

    public Tensor add(Tensor<None>... input) {
        return new TensorOparetor("Add", input) {

            @Operator
            public None compute() {
                IntStream intStream = IntStream.range(0, getInput().length).parallel();
                double value = intStream.mapToDouble(i -> ((None) getInput(i)).getValue()).sum();
                return new None(value);
            }

            public void gradient() {
                None out = (None) getOutput();
                IntStream intStream = IntStream.range(0, getInput().length).parallel();
                intStream.forEach(i -> ((None) getInput(i)).setGrad(out.getGrad()));
            }

        };
    }

    public Tensor addx(Tensor... input) {
        return new TensorFunction("Addx", input) {

            public Object compute() {
                Object A = getInput(0), B = getInput(1), C = shape(Tensor.class, A);
                forEach(A, B, C, (a, b, c, i) -> {
                    c[i] = add(a, b);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor minus(Tensor<None>... input) {
        return new TensorOparetor("Minus", input) {

            @Operator
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
                    None inx = getInput(0), out = (None) getOutput();
                    double grad = out.getGrad();
                    inx.setGrad(-grad);
                } else {
                    None inx = getInput(0), iny = getInput(1), out = (None) getOutput();
                    double grad = out.getGrad();
                    inx.setGrad(grad);
                    iny.setGrad(-grad);
                }
            }

        };
    }

    public Tensor mul(Tensor<None>... input) {
        return new TensorOparetor("Mul", input) {

            @Operator
            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(valx * valy);
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = (None) getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad * valy);
                iny.setGrad(grad * valx);
            }

        };
    }

    public Tensor div(Tensor<None>... input) {
        return new TensorOparetor("Div", input) {

            @Operator
            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(valx / valy);
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = (None) getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad / valy);
                iny.setGrad(-grad * valx / (valy * valy));
            }

        };
    }

    public Tensor exp(Tensor<None>... input) {
        return new TensorOparetor("Exp", input) {

            @Operator
            public None compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return new None(Math.exp(valx));
            }

            public void gradient() {
                None inx = getInput(0), out = (None) getOutput();
                double valx = inx.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad * Math.exp(valx));
            }

        };
    }

    public Tensor expx(Tensor input) {
        return new TensorFunction("Expx", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = exp(a);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor pow(Tensor<None>... input) {
        return new TensorOparetor("Pow", input) {

            @Operator
            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(Math.pow(valx, valy));
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = (None) getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad * valy * Math.pow(valx, valy - 1));
            }

        };
    }

    public Tensor log(Tensor<None>... input) {
        return new TensorOparetor("Log", input) {

            @Operator
            public None compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return new None(Math.log(valx));
            }

            public void gradient() {
                None inx = getInput(0), out = (None) getOutput();
                double valx = inx.getValue();
                double grad = out.getGrad();
                inx.setGrad(grad / valx);
            }

        };
    }

    public Tensor relu(Tensor<None> input) {
        return new TensorOparetor("Relu", input) {

            @Operator
            public None compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return new None(valx > 0 ? valx : 0);
            }

            public void gradient() {
                None inx = getInput(0), out = (None) getOutput();
                double valx = inx.getValue();
                double grad = out.getGrad();
                inx.setGrad(valx > 0 ? grad : 0);
            }

        };
    }

    public Tensor relux(Tensor input) {
        return new TensorFunction("Relux", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = relu(a);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor max(Tensor<None>... input) {
        return new TensorOparetor("Max", input) {

            @Operator
            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return new None(Math.max(valx, valy));
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = (None) getOutput();
                double valx = inx.getValue(), valy = iny.getValue();
                double grad = out.getGrad();
                inx.setGrad(valx > valy ? grad : 0);
                iny.setGrad(valx < valy ? grad : 0);
            }

        };
    }

    public Tensor matmul(Tensor<Tensor[][]>... input) {
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
                None[][] C = (None[][]) getOutput();
                forEach(A.length, B[0].length, A[0].length, (i, l, j) -> {
                    None inx = A[i][j], iny = B[j][l], out = C[i][l];
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setGrad(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor matmulTran(Tensor<Tensor[][]>... input) {
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
                None[][] C = (None[][]) getOutput();
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
                Object C = shape(Tensor.class, B);
                reshape(A, C);
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor prod(Tensor... input) {
        return new TensorFunction("Prod", input) {

            public Object compute() {
                Object A = getInput(0), C = shape(Tensor.class, A);
                Tensor b = getInput(1);
                forEach(A, C, (a, c, i) -> {
                    c[i] = mul(a, b);
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoid(Tensor<Tensor> input) {
        return new TensorFunction("Sigmoid", input) {

            public Tensor compute() {
                Tensor A = getInput(0);
                return div(new TensorConst(1d), add(new TensorConst(1d), exp(minus(A))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidx(Tensor<None> input) {
        return new TensorFunction("Sigmoidx", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = sigmoid(a);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor square(Tensor<None>... input) {
        return new TensorFunction("Square", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return mul(new TensorConst(0.5), pow(minus(a, b), new TensorConst(2d)));
            }

            public void gradient() { }

        };
    }

    public Tensor squarex(Tensor<None>... input) {
        return new TensorFunction("Squarex", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new TensorConst(0d)};
                forEach(A, B, (Func2<Tensor, Tensor>) (a, b) -> {
                    C[0] = add(C[0], square(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor softmaxCross(Tensor<None>... input) {
        return new TensorFunction("SoftmaxCross", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return minus(mul(a, log(b)));
            }

            public void gradient() { }

        };
    }

    public Tensor softmaxCrossx(Tensor<None>... input) {
        return new TensorFunction("SoftmaxCrossx", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new TensorConst(0d)};
                forEach(A, B, (Func2<Tensor, Tensor>) (a, b) -> {
                    C[0] = add(C[0], softmaxCross(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidCross(Tensor<None>... input) {
        return new TensorFunction("SigmoidCross", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return minus(add(mul(a, log(b)), mul(minus(new TensorConst(1), a), log(minus(new TensorConst(1), b)))));
            }

            public void gradient() { }

        };
    }

    public Tensor sigmoidCrossx(Tensor<None>... input) {
        return new TensorFunction("SigmoidCrossx", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new TensorConst(0d)};
                forEach(A, B, (Func2<Tensor, Tensor>) (a, b) -> {
                    C[0] = add(C[0], sigmoidCross(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor sum(Tensor<None> input) {
        return new TensorFunction("Sum", input) {

            public Tensor compute() {
                Object A = getInput(0);
                Tensor[] B = {new TensorConst(0d)};
                forEach(A, a -> {
                    B[0] = add((Tensor) a, B[0]);
                });
                return B[0];
            }

            public void gradient() { }

        };
    }

    public Tensor conv(Tensor<None[][]>... input) {
        return new TensorOparetor("Conv", input) {

            public None[][] compute() {
                None[][] A = getInput(0), B = getInput(1);
                int height = B.length - A.length + 1, width = B[0].length - A[0].length + 1;
                None[][] C = zeroNones(new None[height][width]);
                forEach(height, width, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h + m][w + n], out = C[h][w];
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                None[][] A = getInput(0), B = getInput(1);
                None[][] C = (None[][]) getOutput();
                forEach(C.length, C[0].length, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h + m][w + n], out = C[h][w];
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setValue(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor convx(Tensor... input) {
        return new TensorFunction("Convx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0), B = getInput(1);
                int height = B[0].length - A[0].length + 1, width = B[0][0].length - A[0][0].length + 1;
                Tensor[] C = zeroTensors(new Tensor[A.length], new int[]{height, width});
                forEach(B.length, A.length, (i, l) -> {
                    C[l] = addx(C[l], conv(new Tensor(A[l]), new Tensor(B[i])));
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor deconv(Tensor<None[][]>... input) {
        return new TensorOparetor("Deconv", input) {

            public None[][] compute() {
                None[][] A = getInput(0), B = getInput(1);
                int height = B.length + A.length - 1, width = B[0].length + A[0].length - 1;
                None[][] C = zeroNones(new None[height][width]);
                forEach(B.length, B[0].length, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h][w], out = C[h + m][w + n];
                    out.setValue(out.getValue() + inx.getValue() * iny.getValue());
                });
                return C;
            }

            public void gradient() {
                None[][] A = getInput(0), B = getInput(1);
                None[][] C = (None[][]) getOutput();
                forEach(B.length, B[0].length, A.length, A[0].length, (h, w, m, n) -> {
                    None inx = A[m][n], iny = B[h][w], out = C[h + m][w + n];
                    inx.setGrad(out.getGrad() * iny.getValue());
                    iny.setValue(out.getGrad() * inx.getValue());
                });
            }

        };
    }

    public Tensor deconvx(Tensor... input) {
        return new TensorFunction("Deconvx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0), B = getInput(1);
                int height = B[0].length + A[0].length - 1, width = B[0][0].length + A[0][0].length - 1;
                Tensor[] C = zeroTensors(new Tensor[A.length], new int[]{height, width});
                forEach(B.length, A.length, (i, l) -> {
                    C[l] = addx(C[l], deconv(new Tensor(A[l]), new Tensor(B[i])));
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor maxpool(Tensor input) {
        return new TensorFunction("Maxpool", input) {

            public Tensor[][] compute() {
                Tensor[][] A = getInput(0);
                int height = (int) Math.ceil(A.length / 2.0), width = (int) Math.ceil(A[0].length / 2.0);
                Tensor[][] B = zeroTensors(new Tensor[height][width]);
                forEach(A.length, A[0].length, (y, x) -> {
                    B[y / 2][x / 2] = max(B[y / 2][x / 2], A[y][x]);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor maxpoolx(Tensor input) {
        return new TensorFunction("Maxpoolx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0);
                Tensor[] B = new Tensor[A.length];
                forEach(A.length, i -> {
                    B[i] = maxpool(new Tensor(A[i]));
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor demaxpool(Tensor input) {
        return new TensorFunction("Demaxpool", input) {

            public Tensor[][] compute() {
                Tensor[][] A = getInput(0);
                int height = A.length * 2, width = A[0].length * 2;
                Tensor[][] B = zeroTensors(new Tensor[height][width]);
                forEach(height, width, (y, x) -> {
                    B[y][x] = max(B[y][x], A[y / 2][x / 2]);
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor demaxpoolx(Tensor input) {
        return new TensorFunction("Demaxpoolx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0);
                Tensor[] B = new Tensor[A.length];
                forEach(A.length, i -> {
                    B[i] = demaxpool(new Tensor(A[i]));
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor softmax(Tensor input) {
        return new TensorFunction("Softmax", input) {

            public Object compute() {
                Object[] A = getInput(0);
                Object B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
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

}
