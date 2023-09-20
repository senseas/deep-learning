package com.deep.framework.functions;

import com.deep.framework.graph.None;
import com.deep.framework.graph.ScalarOperator;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.annotation.Cuda;

import static com.deep.framework.lang.ForEach.forEach;
import static java.lang.Math.atan;

public interface Operator {

    default Tensor add(Tensor... input) {
        return new ScalarOperator("Add", input) {

            @Cuda
            public double compute() {
                return inputStream().mapToDouble(a -> ((None) a).getValue()).sum();
            }

            public void gradient(double grad) {
                inputStream().forEach(a -> ((None) a).setGrad(grad));
            }

        };
    }

    default Tensor minus(Tensor... input) {
        return new ScalarOperator("Minus", input) {

            public double compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return valx - valy;
            }

            public void gradient(double grad) {
                None inx = getInput(0), iny = getInput(1);
                inx.setGrad(grad);
                iny.setGrad(-grad);
            }

        };
    }


    default Tensor minus(Tensor input) {
        return new ScalarOperator("Minusx", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return -valx;
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                inx.setGrad(-grad);
            }

        };
    }

    default Tensor mul(Tensor... input) {
        return new ScalarOperator("Mul", input) {

            public double compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return valx * valy;
            }

            public void gradient(double grad) {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(grad * valy);
                iny.setGrad(grad * valx);
            }

        };
    }

    default Tensor div(Tensor... input) {
        return new ScalarOperator("Div", input) {

            public double compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return valx / valy;
            }

            public void gradient(double grad) {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(grad / valy);
                iny.setGrad(-grad * valx / Math.pow(valy, 2));
            }

        };
    }

    default Tensor exp(Tensor... input) {
        return new ScalarOperator("Exp", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.exp(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.exp(valx));
            }

        };
    }

    default Tensor pow(Tensor... input) {
        return new ScalarOperator("Pow", input) {

            public double compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return Math.pow(valx, valy);
            }

            public void gradient(double grad) {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(grad * valy * Math.pow(valx, valy - 1));
            }

        };
    }

    default Tensor log(Tensor... input) {
        return new ScalarOperator("Log", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.log(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / valx);
            }

        };
    }

    default Tensor sum(Tensor input) {
        return new ScalarOperator("Sum", input) {

            public double compute() {
                Object A = getInput(0);
                None B = new None(0d);
                forEach(A, (None a) -> B.setValue(B.getValue() + a.getValue()));
                return B.getValue();
            }

            public void gradient(double grad) {
                Object A = getInput(0);
                forEach(A, (None a) -> a.setGrad(grad));
            }

        };
    }

    default Tensor sin(Tensor... input) {
        return new ScalarOperator("Sin", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.sin(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.cos(valx));
            }

        };
    }

    default Tensor cos(Tensor... input) {
        return new ScalarOperator("Cos", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.cos(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * -Math.sin(valx));
            }

        };
    }

    default Tensor tan(Tensor... input) {
        return new ScalarOperator("Tan", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.tan(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.pow(1 / Math.cos(valx), 2));
            }

        };
    }

    default Tensor cot(Tensor... input) {
        return new ScalarOperator("Cot", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.cos(valx) / Math.sin(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * -Math.pow(1 / Math.sin(valx), 2));
            }

        };
    }

    default Tensor sec(Tensor... input) {
        return new ScalarOperator("Sec", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return 1 / Math.cos(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * Math.tan(valx) / Math.cos(valx));
            }

        };
    }

    default Tensor csc(Tensor... input) {
        return new ScalarOperator("Csc", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return 1 / Math.sin(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad * -Math.cos(valx) / Math.pow(Math.sin(valx), 2));
            }

        };
    }

    default Tensor arcsin(Tensor... input) {
        return new ScalarOperator("Arcsin", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.asin(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / Math.pow(1 - Math.pow(valx, 2), -2));
            }

        };
    }

    default Tensor arccos(Tensor... input) {
        return new ScalarOperator("Arccos", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.acos(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / -Math.pow(1 - Math.pow(valx, 2), -2));
            }

        };
    }

    default Tensor arctan(Tensor... input) {
        return new ScalarOperator("Arctan", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return Math.atan(valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / (1 + Math.pow(valx, 2)));
            }

        };
    }

    default Tensor arccot(Tensor... input) {
        return new ScalarOperator("Arccot", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return atan(1 / valx);
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(grad / -(1 + Math.pow(valx, 2)));
            }

        };
    }

    default Tensor relu(Tensor input) {
        return new ScalarOperator("Relu", input) {

            public double compute() {
                None inx = getInput(0);
                double valx = inx.getValue();
                return valx > 0 ? valx : 0.1 * valx;
            }

            public void gradient(double grad) {
                None inx = getInput(0);
                double valx = inx.getValue();
                inx.setGrad(valx > 0 ? grad : 0.1 * grad);
            }

        };
    }

    default Tensor max(Tensor... input) {
        return new ScalarOperator("Max", input) {

            public double compute() {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                return Math.max(valx, valy);
            }

            public void gradient(double grad) {
                None inx = getInput(0), iny = getInput(1);
                double valx = inx.getValue(), valy = iny.getValue();
                inx.setGrad(valx > valy ? grad : 0);
                iny.setGrad(valx < valy ? grad : 0);
            }

        };
    }

}