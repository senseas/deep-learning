package com.deep.framework.lang.flow;


public interface Operator extends Context, Application {

    default Function add(Function func1, Function func2) {
        double valx = func1.getValue(), valy = func2.getValue();
        double value = valx + valy;

        Gradient grad1 = grad -> func1.setGrad(grad);
        Gradient grad2 = grad -> func2.setGrad(grad);
        return () -> new AppContext(this, value, grad1, grad2);
    }

    default Function minus(Function func1, Function func2) {
        double valx = func1.getValue(), valy = func2.getValue();
        double value = valx - valy;

        Gradient grad1 = grad -> func1.setGrad(grad);
        Gradient grad2 = grad -> func2.setGrad(-grad);
        return () -> new AppContext(this, value, grad1, grad2);
    }

    default Function minus(Function func1) {
        double valx = func1.getValue();

        Gradient grad1 = grad -> func1.setGrad(grad);
        return () -> new AppContext(this, -valx, grad1);
    }

    default Function mul(Function func1, Function func2) {
        double valx = func1.getValue(), valy = func2.getValue();
        double value = valx * valy;

        Gradient grad1 = grad -> func1.setGrad(grad * valy);
        Gradient grad2 = grad -> func2.setGrad(grad * valx);
        return () -> new AppContext(this, value, grad1, grad2);
    }

    default Function div(Function func1, Function func2) {
        double valx = func1.getValue(), valy = func2.getValue();
        double value = valx / valy;

        Gradient grad1 = grad -> func1.setGrad(grad / valy);
        Gradient grad2 = grad -> func2.setGrad(grad * -valx / Math.pow(valy, 2));
        return () -> new AppContext(this, value, grad1, grad2);
    }

    default Function exp(Function func1) {
        double valx = func1.getValue();
        double value = Math.exp(valx);

        Gradient grad1 = grad -> func1.setGrad(grad * value);
        return () -> new AppContext(this, value, grad1);
    }

    default Function pow(Function func1, Function func2) {
        double valx = func1.getValue(), valy = func2.getValue();
        double value = Math.pow(valx, valy);

        Gradient grad1 = grad -> func1.setGrad(grad * valy * Math.pow(valx, valy - 1));
        return () -> new AppContext(this, value, grad1);
    }

    default Function log(Function func1) {
        double valx = func1.getValue();
        double value = Math.log(valx);

        Gradient grad1 = grad -> func1.setGrad(grad / valx);
        return () -> new AppContext(this, value, grad1);
    }

    default Function sin(Function func1) {
        double valx = func1.getValue();
        double value = Math.sin(valx);

        Gradient grad1 = grad -> func1.setGrad(grad * Math.cos(valx));
        return () -> new AppContext(this, value, grad1);
    }

    default Function cos(Function func1) {
        double valx = func1.getValue();
        double value = Math.cos(valx);

        Gradient grad1 = grad -> func1.setGrad(grad * -Math.sin(valx));
        return () -> new AppContext(this, value, grad1);
    }

    default Function tan(Function func1) {
        double valx = func1.getValue();
        double value = Math.tan(valx);

        Gradient grad1 = grad -> func1.setGrad(grad * Math.pow(1 / Math.cos(valx), 2));
        return () -> new AppContext(this, value, grad1);
    }

    default Function cot(Function func1) {
        double valx = func1.getValue();
        double value = Math.cos(valx) / Math.sin(valx);

        Gradient grad1 = grad -> func1.setGrad(grad * -Math.pow(1 / Math.sin(valx), 2));
        return () -> new AppContext(this, value, grad1);
    }

    default Function sec(Function func1) {
        double valx = func1.getValue();
        double value = 1 / Math.cos(valx);

        Gradient grad1 = grad -> func1.setGrad(grad * Math.tan(valx) / Math.cos(valx));
        return () -> new AppContext(this, value, grad1);
    }

    default Function csc(Function func1) {
        double valx = func1.getValue();
        double value = 1 / Math.sin(valx);

        Gradient grad1 = grad -> func1.setGrad(grad * -Math.cos(valx) / Math.pow(Math.sin(valx), 2));
        return () -> new AppContext(this, value, grad1);
    }

    default Function arcsin(Function func1) {
        double valx = func1.getValue();
        double value = Math.asin(valx);

        Gradient grad1 = grad -> func1.setGrad(grad / Math.pow(1 - Math.pow(valx, 2), -2));
        return () -> new AppContext(this, value, grad1);
    }

    default Function arccos(Function func1) {
        double valx = func1.getValue();
        double value = Math.acos(valx);

        Gradient grad1 = grad -> func1.setGrad(grad / -Math.pow(1 - Math.pow(valx, 2), -2));
        return () -> new AppContext(this, value, grad1);
    }

    default Function arctan(Function func1) {
        double valx = func1.getValue();
        double value = Math.atan(valx);

        Gradient grad1 = grad -> func1.setGrad(grad / (1 + Math.pow(valx, 2)));
        return () -> new AppContext(this, value, grad1);
    }

    default Function arccot(Function func1) {
        double valx = func1.getValue();
        double value = Math.atan(1 / valx);

        Gradient grad1 = grad -> func1.setGrad(grad / -(1 + Math.pow(valx, 2)));
        return () -> new AppContext(this, value, grad1);
    }

    default Function relu(Function func1) {
        double valx = func1.getValue();
        double value = valx > 0 ? valx : 0.1 * valx;

        Gradient grad1 = grad -> func1.setGrad(grad * (valx > 0 ? 1 : 0.1));
        return () -> new AppContext(this, value, grad1);
    }

    default Function max(Function func1, Function func2) {
        double valx = func1.getValue(), valy = func2.getValue();
        double value = Math.max(valx, valy);

        Gradient grad1 = grad -> func1.setGrad(grad * (valx > valy ? 1d : 0d));
        Gradient grad2 = grad -> func2.setGrad(grad * valx < valy ? 1d : 0d);
        return () -> new AppContext(this, value, grad1, grad2);
    }

    default Function min(Function func1, Function func2) {
        double valx = func1.getValue(), valy = func2.getValue();
        double value = Math.min(valx, valy);

        Gradient grad1 = grad -> func1.setGrad(grad * (valx < valy ? 1d : 0d));
        Gradient grad2 = grad -> func2.setGrad(grad * (valx > valy ? 1d : 0d));
        return () -> new AppContext(this, value, grad1, grad2);
    }

    default Function var(double d) {
        return () -> new AppContext(this, d);
    }

    default Function one(double d) {
        return () -> new AppContext(this, d);
    }
}