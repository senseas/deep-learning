package com.deep.framework.functions;

import com.deep.framework.graph.ScalarOperator;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.annotation.Cuda;

import java.util.stream.Stream;

import static java.lang.Math.atan;

public interface Operator {

    default Tensor add(Tensor... input) {
        return new ScalarOperator("Add", input) {

            @Cuda
            public double compute() {
                return Stream.of(getInput()).mapToDouble(Tensor::data).sum();
            }

            public void gradient(double grad) {
                Stream.of(getInput()).forEach(a -> a.grad(grad));
            }

        };
    }

    default Tensor minus(Tensor... input) {
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


    default Tensor minus(Tensor input) {
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

    default Tensor mul(Tensor... input) {
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

    default Tensor div(Tensor... input) {
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

    default Tensor exp(Tensor... input) {
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

    default Tensor pow(Tensor... input) {
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

    default Tensor log(Tensor... input) {
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

    default Tensor sum(Tensor input) {
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

    default Tensor sin(Tensor... input) {
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

    default Tensor cos(Tensor... input) {
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

    default Tensor tan(Tensor... input) {
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

    default Tensor cot(Tensor... input) {
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

    default Tensor sec(Tensor... input) {
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

    default Tensor csc(Tensor... input) {
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

    default Tensor arcsin(Tensor... input) {
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

    default Tensor arccos(Tensor... input) {
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

    default Tensor arctan(Tensor... input) {
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

    default Tensor arccot(Tensor... input) {
        return new ScalarOperator("Arccot", input) {

            public double compute() {
                Tensor inx = getInput(0);
                double valx = inx.data();
                return atan(1 / valx);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0);
                double valx = inx.data();
                inx.grad(grad / -(1 + Math.pow(valx, 2)));
            }

        };
    }

    default Tensor relu(Tensor input) {
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

    default Tensor max(Tensor... input) {
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

    default Tensor min(Tensor... input) {
        return new ScalarOperator("Min", input) {

            public double compute() {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                return Math.min(valx, valy);
            }

            public void gradient(double grad) {
                Tensor inx = getInput(0), iny = getInput(1);
                double valx = inx.data(), valy = iny.data();
                inx.grad(valx < valy ? grad : 0);
                iny.grad(valx > valy ? grad : 0);
            }

        };
    }

}