package com.deep.framework.lang.flow;

import java.util.List;
import java.util.Map;

public interface Context {
    double getValue();

    void setValue(double value);

    double getGrad();

    AppContext setGrad(double grad);

    List<Double> data();

    Map<Integer, Double> gradMap();
}