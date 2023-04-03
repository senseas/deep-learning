package com.deep.framework.lang.function;

import com.deep.framework.graph.Tensor;

import java.util.List;
import java.util.Map;

public interface Functor<M> {

    default Functor<M> one(double d) {
        FunctorImpl<M> functor = new FunctorImpl<>(this);
        data().add(d);
        return functor;
    }

    default Functor<M> add(Functor<M> functor1, Functor<M> functor2) {
        FunctorImpl<M> functor = new FunctorImpl<>(this, functor1, functor2);
        double valx = data().get(functor1.idx());
        double valy = data().get(functor2.idx());
        data().add(valx + valy);

        grad().put(functor1.idx(), 1d);
        grad().put(functor2.idx(), 1d);
        return functor;
    }

    default Functor<M> minus(Functor<M> functor1, Functor<M> functor2) {
        FunctorImpl<M> functor = new FunctorImpl<>(this, functor1, functor2);
        double valx = data().get(functor1.idx());
        double valy = data().get(functor2.idx());
        data().add(valx - valy);

        grad().put(functor1.idx(), 1d);
        grad().put(functor2.idx(), -1d);
        return functor;
    }

    default Functor<M> mul(Functor<M> functor1, Functor<M> functor2) {
        FunctorImpl<M> functor = new FunctorImpl<>(this, functor1, functor2);
        double valx = data().get(functor1.idx());
        double valy = data().get(functor2.idx());
        data().add(valx * valy);

        grad().put(functor1.idx(), valy);
        grad().put(functor2.idx(), valx);
        return functor;
    }

    default Functor<M> div(Functor<M> functor1, Functor<M> functor2) {
        FunctorImpl<M> functor = new FunctorImpl<>(this, functor1, functor2);
        double valx = data().get(functor1.idx());
        double valy = data().get(functor2.idx());
        data().add(valx / valy);

        grad().put(functor1.idx(), 1 / valy);
        grad().put(functor2.idx(), valx / Math.pow(valy, 2));
        return functor;
    }


    default Functor<M> pow(Functor<M> functor1, Functor<M> functor2) {
        FunctorImpl<M> functor = new FunctorImpl<>(this, functor1, functor2);
        double valx = data().get(functor1.idx());
        double valy = data().get(functor2.idx());
        data().add(Math.pow(valx, valy));

        grad().put(functor1.idx(), valy * Math.pow(valx, valy - 1));
        grad().put(functor2.idx(), -0d);
        return functor;
    }

    List<Double> data();

    Map<Integer, Double> grad();

    Functor<M>[] functor();

    int idx();

    default Tensor compute() {
        Functor<M> mul = mul(one(0.5), pow(minus(one(3d), one(1d)), one(2d)));
        grad().put(mul.idx(), 1d);
        gradient(mul);
        return null;
    }

    default Tensor gradient(Functor<M> functor) {
        double gradVal = grad().get(functor.idx());
        for (Functor<M> func : functor.functor()) {
            int idx = func.idx();
            grad().put(idx, gradVal * grad().get(idx));
            gradient(func);
        }
        return null;
    }

    static void main(String[] args) {
        FunctorImpl functor = new FunctorImpl();
        functor.compute();
    }
}


