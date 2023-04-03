package com.deep.framework.lang.function;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FunctorImpl<M> implements Functor<M> {
    private List<Double> data = new ArrayList<>();
    private Map<Integer, Double> grad = new HashMap();
    private Functor[] functor;
    private int idx = 0;

    public FunctorImpl() {
    }

    public FunctorImpl(Functor out, Functor<M>... functor) {
        this.idx = out.data().size();
        this.functor = functor;
    }

    public List<Double> data() {
        return data;
    }

    public Map<Integer, Double> grad() {
        return grad;
    }

    public Functor<M>[] functor() {
        return functor;
    }

    public int idx() {
        return idx;
    }
}


