package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.annotation.Operator;
import com.deep.framework.lang.function.Func2;


public class TensorFlow extends Shape {

    public Tensor add(Tensor<None>... input) {
        return new TensorOparetor("Add", input) {

            @Operator
            public None compute() {
                None inx = getInput(0), iny = getInput(1);
                Double valx = inx.getValue(), valy = iny.getValue();
                return new None(valx + valy);
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                Double grad = out.getGrad();
                inx.setGrad(grad);
                iny.setGrad(grad);
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
                    Double valx = inx.getValue();
                    return new None(-valx);
                } else {
                    None inx = getInput(0), iny = getInput(1);
                    Double valx = inx.getValue(), valy = iny.getValue();
                    return new None(valx - valy);
                }
            }

            public void gradient() {
                if (input.length == 1) {
                    None inx = getInput(0), out = getOutput();
                    Double grad = out.getGrad();
                    inx.setGrad(-grad);
                } else {
                    None inx = getInput(0), iny = getInput(1), out = getOutput();
                    Double grad = out.getGrad();
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
                Double valx = inx.getValue(), valy = iny.getValue();
                return new None(valx * valy);
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                Double valx = inx.getValue(), valy = iny.getValue();
                Double grad = out.getGrad();
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
                Double valx = inx.getValue(), valy = iny.getValue();
                return new None(valx / valy);
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                Double valx = inx.getValue(), valy = iny.getValue();
                Double grad = out.getGrad();
                inx.setGrad(grad * valy / Math.pow(valy, 2));
                iny.setGrad(-grad * valx / Math.pow(valy, 2));
            }

        };
    }

    public Tensor exp(Tensor<None>... input) {
        return new TensorOparetor("Exp", input) {

            @Operator
            public None compute() {
                None inx = getInput(0);
                Double valx = inx.getValue();
                return new None(Math.exp(valx));
            }

            public void gradient() {
                None inx = getInput(0), out = getOutput();
                Double valx = inx.getValue();
                Double grad = out.getGrad();
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
                Double valx = inx.getValue(), valy = iny.getValue();
                return new None(Math.pow(valx, valy));
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1);
                Double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(valy * Math.pow(valx, valy - 1));
            }

        };
    }

    public Tensor log(Tensor<None>... input) {
        return new TensorOparetor("Log", input) {

            @Operator
            public None compute() {
                None inx = getInput(0);
                Double valx = inx.getValue();
                return new None(Math.log(valx));
            }

            public void gradient() {
                None inx = getInput(0), out = getOutput();
                Double valx = inx.getValue();
                Double grad = out.getGrad();
                inx.setGrad(grad * 1 / valx);
            }

        };
    }

    public Tensor relu(Tensor<None> input) {
        return new TensorOparetor("Relu", input) {

            @Operator
            public None compute() {
                None inx = getInput(0);
                Double valx = inx.getValue();
                return new None(valx > 0 ? valx : 0);
            }

            public void gradient() {
                None inx = getInput(0), out = getOutput();
                Double valx = inx.getValue();
                Double grad = out.getGrad();
                inx.setGrad(grad * (valx > 0 ? 1 : 0));
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
                Double valx = inx.getValue(), valy = iny.getValue();
                return new None(Math.max(valx, valy));
            }

            public void gradient() {
                None inx = getInput(0), iny = getInput(1), out = getOutput();
                Double grad = out.getGrad();
                inx.setGrad(grad);
                iny.setGrad(grad);
            }

        };
    }

    public Tensor matmul(Tensor<Tensor[][]>... input) {
        return new TensorFunction("Matmul", input) {

            public Tensor[][] compute() {
                Tensor[][] A = getInput(0), B = getInput(1);
                Tensor[][] C = zeros(new Tensor[A.length][B[0].length]);
                forEach(A.length, B[0].length, A[0].length, (i, l, j) -> {
                    C[i][l] = add(C[i][l], mul(A[i][j], B[j][l]));
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor shape(Tensor... input) {
        return new TensorFunction("Shape", input) {

            public Object compute() {
                Object A = getInput(0),B = getInput(1);
                Object C  = shape(Tensor.class, B);
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
                return div(new Tensor(1d), add(new Tensor(1d), exp(minus(A))));
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
                return mul(new Tensor(0.5), pow(minus(a, b), new Tensor(2d)));
            }

            public void gradient() { }

        };
    }

    public Tensor squarex(Tensor<None>... input) {
        return new TensorFunction("Squarex", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new Tensor(0d)};
                forEach(A, B, (Func2<Tensor, Tensor>) (a, b) -> {
                    C[0] = add(C[0], square(a, b));
                });
                return C[0];
            }

            public void gradient() { }

        };
    }

    public Tensor cross(Tensor<None>... input) {
        return new TensorFunction("Cross", input) {

            public Tensor compute() {
                Tensor a = getInput(0), b = getInput(1);
                return minus(mul(a, log(b)));
            }

            public void gradient() { }

        };
    }

    public Tensor crossx(Tensor<None>... input) {
        return new TensorFunction("Crossx", input) {

            public Tensor compute() {
                Object A = getInput(0), B = getInput(1);
                Tensor[] C = {new Tensor(0d)};
                forEach(A, B, (Func2<Tensor, Tensor>) (a, b) -> {
                    C[0] = add(C[0], cross(a, b));
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
                Tensor[] B = {new Tensor(0d)};
                forEach(A, a -> {
                    B[0] = add((Tensor) a, B[0]);
                });
                return B[0];
            }

            public void gradient() { }

        };
    }

    public Tensor conv(Tensor<None[][]>... input) {
        return new TensorFunction("Conv", input) {

            public Object compute() {
                Tensor[][] A = getInput(0), B = getInput(1);
                int height = B.length - A.length + 1, width = B[0].length - A[0].length + 1;
                Tensor[][] C = zeros(new Tensor[height][width]);
                forEach(height, width, A.length, A[0].length, (h, w, m, n) -> {
                    C[h][w] = add(C[h][w], mul(B[h + m][w + n], A[m][n]));
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor convx(Tensor... input) {
        return new TensorFunction("Convx", input) {

            public Object compute() {
                Tensor[][][] A = getInput(0), B = getInput(1);
                int height = B[0].length - A[0].length + 1, width = B[0][0].length - A[0][0].length + 1;
                Tensor[][][] C = zeros(new Tensor[A.length][height][width]);
                forEach(B.length, A.length, (i, l) -> {
                    Tensor<Tensor[][]> tensor = addx(new Tensor(C[l]), conv(new Tensor(A[l]), new Tensor(B[i])));
                    C[l] = tensor.getFunction();
                });
                return C;
            }

            public void gradient() { }

        };
    }

    public Tensor maxpool(Tensor input) {
        return new TensorFunction("Maxpool", input) {

            public Tensor[][] compute() {
                Tensor<None>[][] A = getInput(0);
                int height = (int) Math.ceil(A.length / 2.0), width = (int) Math.ceil(A[0].length / 2.0);
                Tensor<None>[][] B = zeros(new Tensor[height][width]);
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

            public Tensor[][][] compute() {
                Tensor<None>[][][] A = getInput(0);
                Tensor<None>[][][] B = new Tensor[A.length][][];
                forEach(A.length, i -> {
                    Tensor<Tensor[][]> tensor = maxpool(new Tensor(A[i]));
                    B[i] = tensor.getFunction();
                });
                return B;
            }

            public void gradient() { }

        };
    }

    public Tensor softmax(Tensor input) {
        return new TensorFunction("Softmax", input) {

            public Object compute() {
                Object[] A = getInput(0); Object B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = div(exp(a), sum(expx(new Tensor(A))));
                });
                return B;
            }

            public void gradient() { }

        };
    }

}
