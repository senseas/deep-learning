package com.deep.framework.lang.flow;

public interface GradOperator {

    Gradient AddGrad = (double grad, Context... input) -> {
        input[0].gradient(grad);
        input[1].gradient(grad);
    };

    Gradient MinusGrad = (double grad, Context... input) -> {
        input[0].gradient(grad);
        input[1].gradient(-grad);
    };

    Gradient MinusxGrad = (double grad, Context... input) -> {
        input[0].gradient(-grad);
    };

    Gradient MulGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue(), valy = input[1].getValue();
        input[0].gradient(grad * valy);
        input[1].gradient(grad * valx);
    };

    Gradient DivGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue(), valy = input[1].getValue();
        input[0].gradient(grad / valy);
        input[1].gradient(grad * -valx / Math.pow(valy, 2));
    };

    Gradient ExpGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * Math.exp(valx));
    };

    Gradient PowGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue(), valy = input[1].getValue();
        input[0].gradient(grad * valy * Math.pow(valx, valy - 1));
    };

    Gradient LogGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad / valx);
    };

    Gradient SinGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * Math.cos(valx));
    };

    Gradient CosGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * -Math.sin(valx));
    };

    Gradient TanGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * Math.pow(1 / Math.cos(valx), 2));
    };

    Gradient CotGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * -Math.pow(1 / Math.sin(valx), 2));
    };

    Gradient SecGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * Math.tan(valx) / Math.cos(valx));
    };

    Gradient CscGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * -Math.cos(valx) / Math.pow(Math.sin(valx), 2));
    };

    Gradient ArcsinGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad / Math.pow(1 - Math.pow(valx, 2), -2));
    };

    Gradient ArccosGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad / -Math.pow(1 - Math.pow(valx, 2), -2));
    };

    Gradient ArctanGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad / (1 + Math.pow(valx, 2)));
    };

    Gradient ArccotGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad / -(1 + Math.pow(valx, 2)));
    };

    Gradient ReluGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue();
        input[0].gradient(grad * (valx > 0 ? 1 : 0.1));
    };

    Gradient MaxGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue(), valy = input[1].getValue();
        input[0].gradient(grad * (valx > valy ? 1d : 0d));
        input[1].gradient(grad * valx < valy ? 1d : 0d);
    };

    Gradient MinGrad = (double grad, Context... input) -> {
        double valx = input[0].getValue(), valy = input[1].getValue();
        input[0].gradient(grad * (valx < valy ? 1d : 0d));
        input[1].gradient(grad * (valx > valy ? 1d : 0d));
    };

}