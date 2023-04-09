package com.deep.framework.lang.flow;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public interface Context {
    double getValue();

    void setValue(double value);

    List<Consumer> getGradFunc();

    void setGradFunc(Consumer func);

    double getGrad();

    void setGrad(double grad);

    List<Double> data();

    Map<Integer, Double> gradMap();
}