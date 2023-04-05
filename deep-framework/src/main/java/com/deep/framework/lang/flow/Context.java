package com.deep.framework.lang.flow;

import java.util.List;
import java.util.Map;

public interface Context {
    double getValue();

    void setValue(double value);

    double getGradx();

    void setGradx(double gradx);

    double getGrad();

    void setGrad(double grad);

    List<Double> data();

    Map<Integer, Double> gradMap();
}