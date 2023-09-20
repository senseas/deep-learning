package com.deep.framework.graph;

import com.deep.framework.core.TensorFlux;
import com.deep.framework.functions.Operator;
import com.deep.framework.lang.util.BeanUtil;

import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

public class TensorFunction extends Tensor implements Operator {

    public TensorFunction(String name, Tensor... input) {
        super(name, input);
    }

    public <M> M compute() {
        return null;
    }

    public <M> M getInput(int i) {
        Tensor input = getInput()[i];
        if (BeanUtil.isFunction(input)) return (M) input.getFunction();
        return TensorFlux.getTensor(input.getOutput());
    }

    public <M> M getOutput() {
        return (M) output;
    }

    public Object getFunction() {
        if (Objects.nonNull(function)) return function;
        output = null;
        return function = compute();
    }

    public void forward() {
        for (Tensor o : getInput()) o.forward();

        clearOutput();
        forEach(getFunction(), Tensor::forward);

        Object nones = TensorFlux.getOutput(getFunction());
        create(nones);
        forEach(getOutput(), nones, (None out, None none) -> {
            out.setId(none.getId());
            out.setValue(none.getValue());
        });
    }

    public void backward() {
        Object nones = TensorFlux.getOutput(getFunction());
        forEach(getOutput(), nones, (None out, None none) -> {
            none.setGrad(out.getGrad());
        });

        forEach(getFunction(), Tensor::backward);
        forEach(getOutput(), None::reset);
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        forEach(getFunction(), Tensor::reduce);
        for (Tensor o : getInput()) o.reduce();
    }

    public void clearOutput() {
        if (Objects.isNull(value)) return;
        Arrays.fill(value, 0d);
        Arrays.fill(grad, 0d);
    }

    public void create(Object nones) {
        if (Objects.isNull(value)) {
            shape = shapes(nones);
            this.value = random(shape);
            this.grad = zeros(shape);
            this.output = fillNones(this);
        }
    }

}