package com.deep.framework.graph;

import com.deep.framework.bean.None;
import com.deep.framework.lang.util.BeanUtil;

public class Builder extends Shape {

    public static void create(Tenser tenser) {
        if (BeanUtil.isNotOperation(tenser)) {
            Object function = tenser.compute();
            if (BeanUtil.isNotTenser(function)) {
                Tenser tense = (Tenser) function;
                if (BeanUtil.isNotOperation(tense)) {
                    tenser.setFunction(tense.getFunction());
                } else {
                    tenser.setFunction(function);
                }
            } else {
                tenser.setFunction(functions(function));
            }
        } else {
            operator(tenser);
        }
    }

    private static void operator(Tenser tenser) {
        None none = new None(0d);
        tenser.setOutput(none);
    }

    public static <M> M build(Tenser tenser, int i) {
        Tenser input = (Tenser) tenser.getInput()[i];
        if (BeanUtil.isOperation(tenser) && BeanUtil.isNone(input)) return (M) input.getOutput();
        if (BeanUtil.isNone(input)) return Shape.tensers(input.getOutput());
        if (BeanUtil.isOperation(input)) return (M) input;
        return (M) input.getFunction();
    }
}
