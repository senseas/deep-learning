package com.deep.framework.lang.flow;

@FunctionalInterface
public interface Function {
    AppContext apply();

    default double getValue() {
        return apply().getValue();
    }

    default AppContext setGrad(double grad) {
        return apply().setGrad(grad);
    }
}