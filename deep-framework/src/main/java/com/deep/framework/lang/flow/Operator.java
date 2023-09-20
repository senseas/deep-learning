package com.deep.framework.lang.flow;


import static com.deep.framework.lang.flow.GradOperator.*;

public interface Operator {

    default Context add(Context inx, Context iny) {
        double valx = inx.getValue(), valy = iny.getValue();
        double value = valx + valy;
        return new Context(AddGrad, value, inx, iny);
    }

    default Context minus(Context inx, Context iny) {
        double valx = inx.getValue(), valy = iny.getValue();
        double value = valx - valy;
        return new Context(MinusGrad, value, inx, iny);
    }

    default Context minus(Context inx) {
        double valx = inx.getValue();
        return new Context(MinusxGrad, -valx, inx);
    }

    default Context mul(Context inx, Context iny) {
        double valx = inx.getValue(), valy = iny.getValue();
        double value = valx * valy;
        return new Context(MulGrad, value, inx, iny);
    }

    default Context div(Context inx, Context iny) {
        double valx = inx.getValue(), valy = iny.getValue();
        double value = valx / valy;

        return new Context(DivGrad, value, inx, iny);
    }

    default Context exp(Context inx) {
        double valx = inx.getValue();
        double value = Math.exp(valx);
        return new Context(ExpGrad, value, inx);
    }

    default Context pow(Context inx, Context iny) {
        double valx = inx.getValue(), valy = iny.getValue();
        double value = Math.pow(valx, valy);
        return new Context(PowGrad, value, inx, iny);
    }

    default Context log(Context inx) {
        double valx = inx.getValue();
        double value = Math.log(valx);

        return new Context(LogGrad, value, inx);
    }

    default Context sin(Context inx) {
        double valx = inx.getValue();
        double value = Math.sin(valx);

        return new Context(SinGrad, value, inx);
    }

    default Context cos(Context inx) {
        double valx = inx.getValue();
        double value = Math.cos(valx);

        return new Context(CosGrad, value, inx);
    }

    default Context tan(Context inx) {
        double valx = inx.getValue();
        double value = Math.tan(valx);

        return new Context(TanGrad, value, inx);
    }

    default Context cot(Context inx) {
        double valx = inx.getValue();
        double value = Math.cos(valx) / Math.sin(valx);

        return new Context(CotGrad, value, inx);
    }

    default Context sec(Context inx) {
        double valx = inx.getValue();
        double value = 1 / Math.cos(valx);

        return new Context(SecGrad, value, inx);
    }

    default Context csc(Context inx) {
        double valx = inx.getValue();
        double value = 1 / Math.sin(valx);

        return new Context(CscGrad, value, inx);
    }

    default Context arcsin(Context inx) {
        double valx = inx.getValue();
        double value = Math.asin(valx);

        return new Context(ArcsinGrad, value, inx);
    }

    default Context arccos(Context inx) {
        double valx = inx.getValue();
        double value = Math.acos(valx);

        return new Context(ArccosGrad, value, inx);
    }

    default Context arctan(Context inx) {
        double valx = inx.getValue();
        double value = Math.atan(valx);

        return new Context(ArctanGrad, value, inx);
    }

    default Context arccot(Context inx) {
        double valx = inx.getValue();
        double value = Math.atan(1 / valx);

        return new Context(ArccotGrad, value, inx);
    }

    default Context relu(Context inx) {
        double valx = inx.getValue();
        double value = valx > 0 ? valx : 0.1 * valx;

        return new Context(ReluGrad, value, inx);
    }

    default Context max(Context inx, Context iny) {
        double valx = inx.getValue(), valy = iny.getValue();
        double value = Math.max(valx, valy);

        return new Context(MaxGrad, value, inx, iny);
    }

    default Context min(Context inx, Context iny) {
        double valx = inx.getValue(), valy = iny.getValue();
        double value = Math.min(valx, valy);

        return new Context(MinusGrad, value, inx, iny);
    }

    default Context var(double d) {
        return new Context(d);
    }

    default Context cons(double d) {
        return new Context(d);
    }
}