package com.deep.framework.lang.flow;

public interface Operator extends Context, Flow {

    default Operator add(Operator oper1, Operator oper2) {
        Function func = new Function(this, oper1, oper2);
        double valx = oper1.getValue();
        double valy = oper2.getValue();
        func.setValue(valx + valy);

        oper1.setGradx(1d);
        oper2.setGradx(1d);
        return func;
    }

    default Operator minus(Operator oper1, Operator oper2) {
        Function func = new Function(this, oper1, oper2);
        double valx = oper1.getValue();
        double valy = oper2.getValue();
        func.setValue(valx - valy);

        oper1.setGradx(1d);
        oper2.setGradx(-1d);
        return func;
    }

    default Operator minus(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(-valx);

        oper1.setGradx(-1d);
        return func;
    }

    default Operator mul(Operator oper1, Operator oper2) {
        Function func = new Function(this, oper1, oper2);
        double valx = oper1.getValue();
        double valy = oper2.getValue();
        func.setValue(valx * valy);

        oper1.setGradx(valy);
        oper2.setGradx(valx);
        return func;
    }

    default Operator div(Operator oper1, Operator oper2) {
        Function func = new Function(this, oper1, oper2);
        double valx = oper1.getValue();
        double valy = oper2.getValue();
        func.setValue(valx / valy);

        oper1.setGradx(1 / valy);
        oper2.setGradx(-valx / Math.pow(valy, 2));
        return func;
    }

    default Operator exp(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        double exp = Math.exp(valx);
        func.setValue(exp);

        oper1.setGradx(exp);
        return func;
    }

    default Operator pow(Operator oper1, Operator oper2) {
        Function func = new Function(this, oper1, oper2);
        double valx = oper1.getValue();
        double valy = oper2.getValue();
        func.setValue(Math.pow(valx, valy));

        oper1.setGradx(valy * Math.pow(valx, valy - 1));
        oper2.setGradx(-0d);
        return func;
    }

    default Operator log(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.log(valx));

        oper1.setGradx(1 / valx);
        return func;
    }

    default Operator sin(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.sin(valx));

        oper1.setGradx(Math.cos(valx));
        return func;
    }

    default Operator cos(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.cos(valx));

        oper1.setGradx(-Math.sin(valx));
        return func;
    }

    default Operator tan(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.tan(valx));

        oper1.setGradx(Math.pow(1 / Math.cos(valx), 2));
        return func;
    }

    default Operator cot(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.cos(valx) / Math.sin(valx));

        oper1.setGradx(-Math.pow(1 / Math.sin(valx), 2));
        return func;
    }

    default Operator sec(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(1 / Math.cos(valx));

        oper1.setGradx(Math.tan(valx) / Math.cos(valx));
        return func;
    }

    default Operator csc(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(1 / Math.sin(valx));

        oper1.setGradx(-Math.cos(valx) / Math.pow(Math.sin(valx), 2));
        return func;
    }

    default Operator arcsin(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.asin(valx));

        oper1.setGradx(1 / Math.pow(1 - Math.pow(valx, 2), -2));
        return func;
    }

    default Operator arccos(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.acos(valx));

        oper1.setGradx(1 / -Math.pow(1 - Math.pow(valx, 2), -2));
        return func;
    }

    default Operator arctan(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.atan(valx));

        oper1.setGradx(1 / (1 + Math.pow(valx, 2)));
        return func;
    }

    default Operator arccot(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(Math.atan(1 / valx));

        oper1.setGradx(1 / -(1 + Math.pow(valx, 2)));
        return func;
    }

    default Operator relu(Operator oper1) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        func.setValue(valx > 0 ? valx : 0.1 * valx);

        oper1.setGradx(valx > 0 ? 1 : 0.1);
        return func;
    }

    default Operator max(Operator oper1, Operator oper2) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        double valy = oper2.getValue();
        func.setValue(Math.max(valx, valy));

        oper1.setGradx(valx > valy ? 1d : 0d);
        oper2.setGradx(valx < valy ? 1d : 0d);
        return func;
    }

    default Operator min(Operator oper1, Operator oper2) {
        Function func = new Function(this, oper1);
        double valx = oper1.getValue();
        double valy = oper2.getValue();
        func.setValue(Math.min(valx, valy));

        oper1.setGradx(valx < valy ? 1d : 0d);
        oper2.setGradx(valx > valy ? 1d : 0d);
        return func;
    }

    default Operator var(double d) {
        Function func = new Function(this);
        func.setValue(d);
        return func;
    }

    default Operator one(double d) {
        Function func = new Function(this);
        func.setValue(d);
        return func;
    }
}