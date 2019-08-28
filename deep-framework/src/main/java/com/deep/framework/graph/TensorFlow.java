package com.deep.framework.graph;

import com.deep.framework.lang.annotation.Operator;
import com.deep.framework.lang.function.Func2;


public class TensorFlow extends Shape {

    public Tensor add(Node<None>... input) {
        return new Tensor<None>("Add", input) {

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

    public Tensor addx(Node... input) {
        return new Tensor("Addx", input) {

            public Object compute() {
                Object A = getInput(0), B = getInput(1), C = shape(Tensor.class, A);
                forEach(A, B, C, (a, b, c, i) -> {
                    c[i] = add(a, b);
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tensor minus(Node<None>... input) {
        return new Tensor<None>("Minus", input) {

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

    public Tensor mul(Node<None>... input) {
        return new Tensor<None>("Mul", input) {

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

    public Tensor div(Node<None>... input) {
        return new Tensor<None>("Div", input) {

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

    public Tensor exp(Node<None>... input) {
        return new Tensor<None>("Exp", input) {

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

    public Tensor expx(Node input) {
        return new Tensor("Expx", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = exp(a);
                });
                return B;
            }

            public void gradient() {}

        };
    }

    public Tensor pow(Node<None>... input) {
        return new Tensor<None>("Pow", input) {

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

    public Tensor log(Node<None>... input) {
        return new Tensor<None>("Log", input) {

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

    public Tensor relu(Node<None> input) {
        return new Tensor<None>("Relu", input) {

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

    public Tensor relux(Node input) {
        return new Tensor<None>("Relu", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = relu(a);
                });
                return B;
            }

            public void gradient() {}

        };
    }

    public Tensor max(Node<None>... input) {
        return new Tensor<None>("Max", input) {

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

    public Tensor matmul(Node<Node[][]>... input) {
        return new Tensor<Node[][]>("Matmul", input) {

            public Node[][] compute() {
                Node[][] A = getInput(0), B = getInput(1);
                Node[][] C = zeros(new Node[A.length][B[0].length]);
                forEach(A.length, B[0].length, A[0].length, (i, l, j) -> {
                    C[i][l] = add(C[i][l], mul(A[i][j], B[j][l]));
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tensor shape(Node... input) {
        return new Tensor("Shape", input) {

            public Object compute() {
                Tensor A = (Tensor) input[0], B = (Tensor) input[1];
                Object C = Nones(B.getOutput());
                reshape(A.getOutput(), C);
                setOutput(C);
                return new Tensor[0];
            }

            public void gradient() {}

        };
    }

    public Tensor prod(Node... input) {
        return new Tensor("Prod", input) {

            public Object compute() {
                Object A = getInput(0), C = shape(Tensor.class, A);
                Tensor b = (Tensor) getInput(1);
                forEach(A, C, (a, c, i) -> {
                    c[i] = mul(a, b);
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tensor sigmoid(Node<Tensor> input) {
        return new Tensor<Node>("Sigmoid", input) {

            public Node compute() {
                Tensor A = getInput(0);
                return div(new Tensor(1d), add(new Tensor(1d), exp(minus(A))));
            }

            public void gradient() {}

        };
    }

    public Tensor sigmoidx(Node<None> input) {
        return new Tensor("Sigmoidx", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = sigmoid(a);
                });
                return B;
            }

            public void gradient() {}

        };
    }

    public Tensor square(Node<None>... input) {
        return new Tensor<Node>("Square", input) {

            public Node compute() {
                Node a = getInput(0), b = getInput(1);
                return mul(new Tensor(0.5), pow(minus(a, b), new Tensor(2d)));
            }

            public void gradient() {}

        };
    }

    public Tensor squarex(Node<None>... input) {
        return new Tensor("Squarex", input) {

            public Node compute() {
                Object A = getInput(0), B = getInput(1);
                Node[] C = {new Tensor(0d)};
                forEach(A, B, (Func2<Node, Node>) (a, b) -> {
                    C[0] = add(C[0], square(a, b));
                });
                return C[0];
            }

            public void gradient() {}

        };
    }

    public Tensor cross(Node<None>... input) {
        return new Tensor<Node>("Cross", input) {

            public Node compute() {
                Node a = getInput(0), b = getInput(1);
                return minus(mul(a, log(b)));
            }

            public void gradient() {}

        };
    }

    public Tensor crossx(Node<None>... input) {
        return new Tensor("Crossx", input) {

            public Node compute() {
                Object A = getInput(0), B = getInput(1);
                Node[] C = {new Tensor(0d)};
                forEach(A, B, (Func2<Node, Node>) (a, b) -> {
                    C[0] = add(C[0], cross(a, b));
                });
                return C[0];
            }

            public void gradient() {}

        };
    }

    public Tensor sum(Node<None> input) {
        return new Tensor<Node>("Sum", input) {

            public Node compute() {
                Object A = getInput(0);
                Node[] B = {new Tensor(0d)};
                forEach(A, a -> {
                    B[0] = add((Tensor) a, B[0]);
                });
                return B[0];
            }

            public void gradient() {}

        };
    }

    public Tensor conv(Node<None[][]>... input) {
        return new Tensor<Node[][]>("Conv", input) {

            public Object compute() {
                Node[][] A = getInput(0), B = getInput(1);
                int height = B.length - A.length + 1, width = B[0].length - A[0].length + 1;
                Node[][] C = zeros(new Node[height][width]);
                forEach(height, width, A.length, A[0].length, (h, w, m, n) -> {
                    C[h][w] = add(C[h][w], mul(B[h + m][w + n], A[m][n]));
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tensor convx(Node... input) {
        return new Tensor<Node[][][]>("Convx", input) {

            public Object compute() {
                Node[][][] A = getInput(0), B = getInput(1);
                int height = B[0].length - A[0].length + 1, width = B[0][0].length - A[0][0].length + 1;
                Node[][][] C = zeros(new Node[A.length][height][width]);
                forEach(B.length, A.length, (i, l) -> {
                    Tensor<Node[][]> tensor = addx(new Tensor(C[l]), conv(new Tensor(A[l]), new Tensor(B[i])));
                    C[l] = tensor.getFunction();
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tensor maxpool(Node input) {
        return new Tensor<Node>("Maxpool", input) {

            public Node[][] compute() {
                Node<None>[][] A = getInput(0);
                int height = (int) Math.ceil(A.length / 2.0), width = (int) Math.ceil(A[0].length / 2.0);
                Node<None>[][] B = zeros(new Node[height][width]);
                forEach(A.length, A[0].length, (y, x) -> {
                    B[y / 2][x / 2] = max(B[y / 2][x / 2], A[y][x]);
                });
                return B;
            }

            public void gradient() {}

        };
    }

    public Tensor maxpoolx(Node input) {
        return new Tensor<Node>("Maxpoolx", input) {

            public Node[][][] compute() {
                Node<None>[][][] A = getInput(0);
                Node<None>[][][] B = new Node[A.length][][];
                forEach(A.length, i -> {
                    Tensor<Node[][]> tensor = maxpool(new Tensor(A[i]));
                    B[i] = tensor.getFunction();
                });
                return B;
            }

            public void gradient() {}

        };
    }

    public Tensor softmax(Node input) {
        return new Tensor("Softmax", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tensor.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = div(exp(a), sum(expx(input)));
                });
                return B;
            }

            public void gradient() {}

        };
    }

}
