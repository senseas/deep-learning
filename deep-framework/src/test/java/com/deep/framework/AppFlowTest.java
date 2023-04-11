package com.deep.framework;

import com.deep.framework.lang.flow.Function;
import com.deep.framework.lang.flow.AppContext;
import org.junit.Test;

public class AppFlowTest {

    @Test
    public void squareTest() {
        AppContext functor = new AppContext() {

            public void compute() {
                Function mul = mul(one(0.5), pow(minus(one(0.01), var(0.391249035007275)), one(2d)));
                AppContext apply = mul.apply();
                apply.setGrad(1).gradient();
            }

        };
        functor.compute();
    }

}
