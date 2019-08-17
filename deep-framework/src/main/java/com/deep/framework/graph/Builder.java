package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;

public class Builder extends Shape {

    public static <M> M function(Tenser tenser) {
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
            return (M) function;
        }
        return null;
    }

    public static <M> M build(Tenser tenser, int i) {
        Tenser<Tenser> input = (Tenser) tenser.getInput()[i];
        if (BeanUtil.isOperation(tenser) && BeanUtil.isNone(input)) return (M) input.getOutput();
        if (BeanUtil.isOperation(tenser) && BeanUtil.isOperation(input)) return (M) input.getOutput();
        if (BeanUtil.isOperation(tenser)) return (M) input.getFunction().getOutput();
        if (BeanUtil.isOperation(input)) return (M) input;
        return Shape.tensers(input.getOutput());
    }

}
