package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import java.util.Arrays;
import java.util.stream.Collectors;

public interface Operator {

    default Tensor cons(String input) {
        return new Tensor(input);
    }

    default Tensor cons(double input) {
        return new Tensor("Cons", String.valueOf(input));
    }

    default Tensor Tensor(Tenser input) {
        return new TensorFunction(input);
    }

    default Tensor add(Tensor... input) {
        return new TensorOperator("Add", input) {

            public String compute() {
                return Arrays.stream(getInput()).map(a -> a.getOutput().one().getVarId()).collect(Collectors.joining("+"));
            }

            public void gradient(String grad) {
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                inx.setGrad(grad);
                iny.setGrad("-" + grad);
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad("-" + "(" + grad + ")");
            }

        };
    }

    default Tensor mul(Tensor... input) {
        return new TensorOperator("Mul", input) {

            public String compute() {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                return valx + "*" + valy;
            }

            public void gradient(String grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                inx.setGrad(grad + "*" + valy);
                iny.setGrad(grad + "*" + valx);
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                inx.setGrad(grad + "/" + valy);
                iny.setGrad("-" + grad + "*" + valx + "/Math.pow(" + valy + ",2)");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                inx.setGrad(grad + "*" + inx.getVarId());
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                inx.setGrad(grad + "*" + valy + "*Math.pow(" + valx + "," + valy + "-1)");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "/" + valx);
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "*Math.cos(" + valx + ")");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "*-Math.sin(" + valx + ")");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "*Math.pow(1/Math.cos(" + valx + "),2)");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "*-Math.pow(1/Math.sin(" + valx + "),2)");
            }

        };
    }

    default Tensor sec(Tensor... input) {
        return new TensorOperator("Sec", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return "1/" + "Math.cos(" + valx + ")";
            }

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "*Math.tan(" + valx + ")/Math.cos(" + valx + ")");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "*-Math.cos(" + valx + ")/Math.pow(Math.sin(" + valx + "),2)");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "/Math.pow(1-Math.pow(" + valx + ",2),-2)");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "/-Math.pow(1-Math.pow(" + valx + ",2),-2)");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "/(1+Math.pow(" + valx + ",2))");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(grad + "/-(1+Math.pow(" + valx + ",2))");
            }

        };
    }

    default Tensor relu(Tensor input) {
        return new TensorOperator("Relu", input) {

            public String compute() {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                return valx + ">0? " + valx + " : 0.1*" + valx;
            }

            public void gradient(String grad) {
                Tensor inx = getInput(0).one();
                String valx = inx.getVarId();
                inx.setGrad(valx + ">0? " + grad + ":0.1*" + grad);
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                inx.setGrad(valx + ">" + valy + "?" + grad + ":0");
                iny.setGrad(valx + "<" + valy + "?" + grad + ":0");
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

            public void gradient(String grad) {
                Tensor inx = getInput(0).one(), iny = getInput(1).one();
                String valx = inx.getVarId(), valy = iny.getVarId();
                inx.setGrad(valx + "<" + valy + "?" + grad + ":0");
                iny.setGrad(valx + ">" + valy + "?" + grad + ":0");
            }

        };
    }

    default Tensor sum(Tensor input) {
        return new TensorOperator("Sum", input) {

            public String compute() {
                Tenser<Tensor> A = getInput(0);
                return A.stream().map(Tensor::getVarId).collect(Collectors.joining("+"));
            }

            public void gradient(String grad) {
                Tenser<Tensor> A = getInput(0);
                A.stream().forEach(a -> a.setGrad(grad));
            }

        };
    }

}