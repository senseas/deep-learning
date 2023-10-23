package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

import static com.deep.framework.lang.ForEach.forEach;

public class TensorFunction extends Tensor {

    public TensorFunction(String name, int[] shape, Tensor... input) {
        super(name, shape, input);
    }

    public TensorFunction(Tenser<Tensor> function) {
        super(function.one().getName(), function.shape);
        this.function = function;
    }

    public Tenser<Tensor> compute() {
        return null;
    }

    public void forward() {
        if (status) return;
        for (Tensor o : getInput()) o.forward();
        forEach(getFunction(), Tensor::forward);

        Tenser<Tensor> nones = getOutput(getFunction());
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> out.setId(none.getId()));
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> out.setData(none.getData()));
        status = true;
    }

    public void backward() {
        Tenser<Tensor> nones = getOutput(getFunction());
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> none.setGrad(out.getGrad()));

        forEach(getFunction(), Tensor::backward);
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = compute();
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public void clearGrad() {
        forEach(getOutput(), (Tensor out) -> out.setGrad("0d"));
    }

}