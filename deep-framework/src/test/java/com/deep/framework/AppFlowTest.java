package com.deep.framework;

import com.deep.framework.lang.flow.Function;
import com.deep.framework.lang.flow.Operator;
import org.junit.Test;

public class AppFlowTest {

    @Test
    public void squareTest() {
        Function functor = new Function() {

            public void compute() {
                Operator mul = mul(one(0.5), pow(minus(one(0.01), var(0.391249035007275)), one(2d)));
                mul.setGrad(1d);
                mul.gradient();
            }

        };
        functor.compute();
    }

}
