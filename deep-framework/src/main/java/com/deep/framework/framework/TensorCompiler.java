package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFunctor;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static com.deep.framework.lang.ForEach.forEach;

public class TensorCompiler implements Serializable {

    public TensorFunctor add(Tensor... input) {
        return new TensorFunctor("Add") {

            public String compute() {
                return getValId() + "=" + Arrays.stream(getInput())
                .map(a -> (None) a.getOutput()).map(a -> "+" + a.getValId())
                .collect(Collectors.joining()) + ";";
            }

            public String gradient(String grad) {
                return Arrays.stream(getInput()).map(a -> (None) a.getOutput())
                 .filter(None::isGradre).map(a -> a.getGradId() + "=" + getGradId() + ";")
                 .collect(Collectors.joining());
            }

        };
    }

    public TensorFunctor minus(Tensor... input) {
        return new TensorFunctor("Minus", input) {

            public String compute() {
                None inx = getInput(0), iny = getInput(1);
                return getValId() + "=" + inx.getValId() + "-" + iny.getValId() + ";";
            }

            public String gradient(String grad) {
                None inx = getInput(0), iny = getInput(1);
                if (inx.isGradre()) grad += inx.getGradId() + "=" + getGradId() + ";";
                if (iny.isGradre()) grad += iny.getGradId() + "=-" + getGradId() + ";";
                return grad;
            }

        };
    }

    public TensorFunctor minus(Tensor input) {
        return new TensorFunctor("Minusx", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=-" + inx.getValId() + ";";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=-" + getGradId() + ";";
            }

        };
    }

    public TensorFunctor mul(Tensor... input) {
        return new TensorFunctor("Mul", input) {

            public String compute() {
                None inx = getInput(0), iny = getInput(1);
                return getValId() + "=" + inx.getValId() + "*" + iny.getValId() + ";";
            }

            public String gradient(String grad) {
                None inx = getInput(0), iny = getInput(1);
                if (inx.isGradre()) grad += inx.getGradId() + "=" + getGradId() + "*" + iny.getValId() + ";";
                if (iny.isGradre()) grad += iny.getGradId() + "=" + getGradId() + "*" + inx.getValId() + ";";
                return grad;
            }

        };
    }

    public TensorFunctor div(Tensor... input) {
        return new TensorFunctor("Div", input) {

            public String compute() {
                None inx = getInput(0), iny = getInput(1);
                return getValId() + "=" + inx.getValId() + "/" + iny.getValId() + ";";
            }

            public String gradient(String grad) {
                None inx = getInput(0), iny = getInput(1);
                if (inx.isGradre()) grad += inx.getGradId() + "=" + getGradId() + "/" + iny.getValId() + ";";
                if (iny.isGradre()) grad += iny.getGradId() + "=-" + getGradId() + "*" + inx.getValId() + "/pow(" + iny.getValId() + ",2);";
                return grad;
            }

        };
    }

    public TensorFunctor exp(Tensor... input) {
        return new TensorFunctor("Exp", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=exp(" + inx.getValId() + ");";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "*exp(" + inx.getValId() + ");";
            }

        };
    }

    public TensorFunctor pow(Tensor... input) {
        return new TensorFunctor("Pow", input) {

            public String compute() {
                None inx = getInput(0), iny = getInput(1);
                return getValId() + "=pow(" + inx.getValId() + "," + iny.getValId() + ");";
            }

            public String gradient(String grad) {
                None inx = getInput(0), iny = getInput(1);
                return inx.getGradId() + "=" + getGradId() + "*" + iny.getValId() + "*pow(" + inx.getValId() + "," + iny.getValId() + "-1);";
            }

        };
    }

    public TensorFunctor log(Tensor... input) {
        return new TensorFunctor("Log", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=log(" + inx.getValId() + ");";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "/" + inx.getValId() + ";";
            }

        };
    }

    public TensorFunctor sin(Tensor... input) {
        return new TensorFunctor("Sin", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=sin(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "*cos(" + inx.getValId() + ")";
            }

        };
    }

    public TensorFunctor cos(Tensor... input) {
        return new TensorFunctor("Cos", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=sin(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "*-sin(" + inx.getValId() + ")";
            }

        };
    }

    public TensorFunctor tan(Tensor... input) {
        return new TensorFunctor("Tan", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=tan(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "*pow(1/cos(" + inx.getValId() + "),2)";
            }

        };
    }

    public TensorFunctor cot(Tensor... input) {
        return new TensorFunctor("Cot", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=cos(" + inx.getValId() + ")/sin(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "*-pow(1/sin(" + inx.getValId() + "),2))";
            }

        };
    }

    public TensorFunctor sec(Tensor... input) {
        return new TensorFunctor("Sec", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=1/cos(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "*tan(" + inx.getValId() + ")/cos(" + inx.getValId() + ")";
            }

        };
    }

    public TensorFunctor csc(Tensor... input) {
        return new TensorFunctor("Csc", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=1/sin(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "*-cos(" + inx.getValId() + ")/pow(sin(" + inx.getValId() + "),2)";
            }

        };
    }

    public TensorFunctor arcsin(Tensor... input) {
        return new TensorFunctor("Arcsin", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=asin(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "/pow(1-pow(" + inx.getValId() + ",2),-2)";
            }

        };
    }

    public TensorFunctor arccos(Tensor... input) {
        return new TensorFunctor("Arccos", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=acos(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "/-pow(1-pow(" + inx.getValId() + ",2),-2)";
            }

        };
    }

    public TensorFunctor arctan(Tensor... input) {
        return new TensorFunctor("Arctan", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=atan(" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "/(1+pow(" + inx.getValId() + ",2))";
            }

        };
    }

    public TensorFunctor arccot(Tensor... input) {
        return new TensorFunctor("Arccot", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=atan(1/" + inx.getValId() + ")";
            }

            public String gradient(String grad) {
                None inx = getInput(0);
                return inx.getGradId() + "=" + getGradId() + "/-(1+pow(" + inx.getValId() + ",2)))";
            }

        };
    }

    public TensorFunctor relu(Tensor input) {
        return new TensorFunctor("Relu", input) {

            public String compute() {
                None inx = getInput(0);
                return getValId() + "=" + inx.getValId() + "*(" + inx.getValId() + " > 0 ? 1 : 0.1);";
            }

            public String gradient(String gard) {
                None inx = getInput(0);
                return inx.getGradId() +"=" + getGradId() + "*(" + inx.getValId() + " > 0 ? 1 : 0.1);";
            }

        };
    }

    public TensorFunctor sum(Tensor input) {
        return new TensorFunctor("Sum", input) {

            public String compute() {
                Object A = getInput(0);
                List<None> list = new ArrayList<>();
                forEach(A, (None a) -> list.add(a));
                return getValId() + "=" + list.stream().map(a -> "+" + a.getValId()).collect(Collectors.joining()) + ";";
            }

            public String gradient(String grad) {
                Object A = getInput(0);
                List<None> list = new ArrayList<>();
                forEach(A, (None a) -> list.add(a));
                return list.stream().filter(None::isGradre).map(a -> a.getGradId() + "=" + getGradId() + ";").collect(Collectors.joining());
            }

        };
    }

}