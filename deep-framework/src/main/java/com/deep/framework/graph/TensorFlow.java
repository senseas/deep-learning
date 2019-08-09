package com.deep.framework.graph;

import com.deep.framework.bean.Node;
import com.deep.framework.bean.None;
import com.deep.framework.lang.annotation.Operator;


public class TensorFlow extends Shape {

    public Tenser add(Node<None>... input) {
        return new Tenser<None>("Add", input) {

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

    public Tenser addx(Node... input) {
        return new Tenser("Addx", input) {

            public Object compute() {
                Object A = getInput(0), B = getInput(1), C = shape(Tenser.class, A);
                forEach(A, B, C, (a, b, c, i) -> {
                    c[i] = add(a, b);
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tenser minus(Node<None>... input) {
        return new Tenser<None>("Minus", input) {

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

    public Tenser mul(Node<None>... input) {
        return new Tenser<None>("Mul", input) {

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

    public Tenser div(Node<None>... input) {
        return new Tenser<None>("Div", input) {

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

    public Tenser exp(Node<None>... input) {
        return new Tenser<None>("Exp", input) {

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

    public Tenser expx(Node input) {
        return new Tenser("Expx", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tenser.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = exp(a);
                });
                return B;
            }

            public void gradient() {}

        };
    }

    public Tenser pow(Node<None>... input) {
        return new Tenser<None>("Pow", input) {

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

    public Tenser relu(Node<None> input) {
        return new Tenser<None>("Relu", input) {

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

    public Tenser matmul(Node<Node[][]>... input) {
        return new Tenser<Node[][]>("Matmul", input) {

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

    public Tenser prod(Node node) {
        return new Tenser<Node>("Prod", node) {

            public Node compute() {
                return null;
            }

            public void gradient() {}

        };
    }

    public Tenser sigmoid(Node<Tenser> input) {
        return new Tenser<Node>("Sigmoid", input) {

            public Node compute() {
                Tenser A = getInput(0);
                return div(new Tenser(1d), add(new Tenser(1d), exp(minus(A))));
            }

            public void gradient() {}

        };
    }

    public Tenser sigmoidx(Node<None> input) {
        return new Tenser("Sigmoidx", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tenser.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = sigmoid(a);
                });
                return B;
            }

            public void gradient() {}

        };
    }

    public Tenser square(Node<None>... input) {
        return new Tenser<Node>("Square", input) {

            public Node compute() {
                Node a = getInput(0), b = getInput(1);
                return mul(new Tenser(0.5), pow(minus(a, b), new Tenser(2d)));
            }

            public void gradient() {}

        };
    }

    public Tenser squarex(Node<None>... input) {
        return new Tenser("Squarex", input) {

            public Object compute() {
                Object A = getInput(0), B = getInput(1), C = shape(Tenser.class, A);
                forEach(A, B, C, (a, b, c, i) -> {
                    c[i] = square(a, b);
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tenser sum(Node<None> input) {
        return new Tenser<Node>("Sum", input) {

            public Node compute() {
                Object A = getInput(0);
                Node[] B = {new Tenser(0d)};
                forEach(A, a -> {
                    B[0] = add((Tenser) a, B[0]);
                });
                return B[0];
            }

            public void gradient() {}

        };
    }

    public Tenser conv(Node<None[][]>... input) {
        return new Tenser<Node[][]>("Conv", input) {

            public Object compute() {
                Node[][] A = getInput(0), B = getInput(1);
                int height = B.length - A.length + 1, width = B[0].length - A[0].length + 1;
                Node[][] C = zeros(new Node[height][width]);
                forEach(height, width, A.length, A[0].length, (i, l, m, n) -> {
                    C[i][l] = add(C[i][l], mul(B[i + m][l + n], A[m][n]));
                });
                return C;
            }

            public void gradient() {}

        };
    }

    public Tenser maxpool(Node node) {
        return new Tenser<Node>("Maxpool", node) {

            public Node compute() {
                return null;
            }

            public void gradient() {}

        };
    }

    public Tenser softmax(Node input) {
        return new Tenser("Softmax", input) {

            public Object compute() {
                Object A = getInput(0), B = shape(Tenser.class, A);
                forEach(A, B, (a, b, i) -> {
                    b[i] = div(exp(a), sum(expx(input)));
                });
                return B;
            }

            public void gradient() {}

        };
    }

}
