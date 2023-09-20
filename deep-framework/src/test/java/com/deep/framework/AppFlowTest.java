package com.deep.framework;

import com.deep.framework.lang.flow.Function;
import com.deep.framework.lang.flow.Context;
import org.junit.Test;

public class AppFlowTest {

    @Test
    public void squareTest() {
        Function functor = new Function() {

            public void compute() {
                Context mul = mul(cons(0.5), pow(minus(cons(0.01), var(0.391249035007275)), cons(2d)));
                mul.gradient(1);
            }

        };
        functor.compute();
    }

}
