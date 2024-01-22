package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.stream.Collectors;

public interface Operator {

    default Tensor cons(String input) {
        return new TensorConst(input);
    }

    default Tensor cons(double input) {
        return new TensorConst(String.valueOf(input));
    }

    default Tensor Tensor(Tenser input) {
        return new TensorFunction(input);
    }

    default Tensor add(Tensor... input) {
        return new TensorOperator("Add", input) {

            public String compute() {
                return "(" + Arrays.stream(getInput()).map(a -> a.getOutput().one().getVarId()).collect(Collectors.joining("+")) + ")";
            }

            public void gradient(Tensor grad) {
                Arrays.stream(getInput()).map(a -> a.getOutput().one()).forEach(a -> a.setGrad(grad));
            }

        };
    }

    default Tensor minus(Tensor... input) {
        return new TensorOperator("Minus", input) {

            public String compute() {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                return valx + "-" + valy;
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                inx.setGrad(grad);
                iny.setGrad(mul(grad, cons(-1)));
            }

        };
    }

    default Tensor minus(Tensor input) {
        return new TensorOperator("Minusx", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "-" + valx;
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, cons(-1)));
            }

        };
    }

    default Tensor mul(Tensor... input) {
        return new TensorOperator("Mul", input) {

            public String compute() {
                return Arrays.stream(getInput()).map(a -> {
                    Tensor one = a.getOutput().one();
                    String varId = one.getVarId();
                    if (one instanceof TensorConst) return varId;
                    return varId.startsWith("(") && varId.endsWith(")") ? varId : "(" + varId + ")";
                }).collect(Collectors.joining("*"));
            }

            public void gradient(Tensor grad) {
                Arrays.stream(getInput()).forEach(a -> a.setGrad(mul(Arrays.stream(getInput()).map(c -> a.getOutput().one() == c.getOutput().one() ? grad : c.getOutput().one()).toArray(Tensor[]::new))));
            }

        };
    }

    default Tensor div(Tensor... input) {
        return new TensorOperator("Div", input) {

            public String compute() {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                return valx + "/" + valy;
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                inx.setGrad(mul(grad, div(cons(1), iny)));
                iny.setGrad(mul(grad, inx, div(cons(-1), pow(iny, cons(2)))));
            }

        };
    }

    default Tensor exp(Tensor... input) {
        return new TensorOperator("Exp", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.exp(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, exp(inx)));
            }

        };
    }

    default Tensor pow(Tensor... input) {
        return new TensorOperator("Pow", input) {

            public String compute() {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                return "Math.pow(" + valx + "," + valy + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                inx.setGrad(mul(grad, iny, iny.getData().equals("2.0") ? inx : pow(inx, minus(iny, cons(1)))));
            }

        };
    }

    default Tensor log(Tensor... input) {
        return new TensorOperator("Log", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.log(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, div(cons(1), inx)));
            }

        };
    }

    default Tensor sin(Tensor... input) {
        return new TensorOperator("Sin", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.sin(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, cos(inx)));
            }

        };
    }

    default Tensor cos(Tensor... input) {
        return new TensorOperator("Cos", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.cos(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, minus(sin(inx))));
            }

        };
    }

    default Tensor tan(Tensor... input) {
        return new TensorOperator("Tan", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.tan(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, pow(div(cons(1), cos(inx)), cons(2))));
            }

        };
    }

    default Tensor cot(Tensor... input) {
        return new TensorOperator("Cot", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.cos(" + valx + ")/Math.sin(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, minus(pow(div(cons(1)), sin(inx), cons(2)))));
            }

        };
    }

    default Tensor sec(Tensor... input) {
        return new TensorOperator("Sec", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "1/Math.cos(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, div(tan(inx), cos(inx))));
            }

        };
    }

    default Tensor csc(Tensor... input) {
        return new TensorOperator("Csc", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "1/Math.sin(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, minus(div(cos(inx), pow(sin(inx), cons(2))))));
            }

        };
    }

    default Tensor arcsin(Tensor... input) {
        return new TensorOperator("Arcsin", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.asin(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, div(cons(1), pow(minus(cons(1), pow(inx, cons(2))), cons(-2)))));
            }

        };
    }

    default Tensor arccos(Tensor... input) {
        return new TensorOperator("Arccos", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.acos(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, div(cons(-1), pow(minus(cons(1), pow(inx, cons(2))), cons(-2)))));
            }

        };
    }

    default Tensor arctan(Tensor... input) {
        return new TensorOperator("Arctan", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "Math.atan(" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, div(cons(1), add(cons(1), pow(inx, cons(2))))));
            }

        };
    }

    default Tensor arccot(Tensor... input) {
        return new TensorOperator("Arccot", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "atan(1/" + valx + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(mul(grad, div(cons(-1), add(cons(1), pow(inx, cons(2))))));
            }

        };
    }

    default Tensor where(Tensor... input) {
        return new TensorOperator("Max", input) {

            public String compute() {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                Tensor inm = getInput(2).one(), inn = getInput(3).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                return valx + ">" + valy + "?" + inm.getData() + inn.getData();
            }

            public void gradient(Tensor grad) {}

        };
    }

    default Tensor relu(Tensor input) {
        return new TensorOperator("Relu", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return valx + ">0? " + valx + " : 0.1*" + valx;
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(where(inx, cons(0), grad, mul(cons(0.1), grad)));
            }

        };
    }

    default Tensor max(Tensor... input) {
        return new TensorOperator("Max", input) {

            public String compute() {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                return "Math.max(" + valx + "," + valy + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                inx.setGrad(where(inx, iny, grad, cons(0)));
                iny.setGrad(where(iny, inx, grad, cons(0)));
            }

        };
    }

    default Tensor min(Tensor... input) {
        return new TensorOperator("Min", input) {

            public String compute() {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                return "Math.min(" + valx + "," + valy + ")";
            }

            public void gradient(Tensor grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                inx.setGrad(where(iny, inx, grad, cons(0)));
                iny.setGrad(where(inx, iny, grad, cons(0)));
            }

        };
    }

    default Tensor sum(Tensor input) {
        return new TensorOperator("Sum", input) {

            public String compute() {
                Tenser<Tensor> A = getInput(0);
                return A.stream().map(Tensor::getVarId).collect(Collectors.joining("+"));
            }

            public void gradient(Tensor grad) {
                Tenser<Tensor> A = getInput(0);
                A.stream().forEach(a -> a.setGrad(grad));
            }

        };
    }

}