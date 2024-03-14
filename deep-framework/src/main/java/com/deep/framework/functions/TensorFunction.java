package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import java.util.Objects;

import static com.deep.framework.lang.ForEach.forEach;

public class TensorFunction extends Tensor {

    public TensorFunction(String name, int[] shape, Tensor... input) {
        super(name, shape, input);
    }

    public TensorFunction(Tenser<Tensor> function) {
        super(function.data(0).getName(), function.shape);
        this.function = function;
    }

    public Tenser<Tensor> compute() {
        return null;
    }

    public void forward() {
        if (status) return;
        for (Tensor o : getInput()) o.forward();
        getFunction().forEach(Tensor::forward);

        Tenser<Tensor> nones = getOutput(getFunction());
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> out.setId(none.getId()).setData(none.getData()));
        status = true;
    }

    public void backward() {
        if (states) return;
        for (Tensor o : getInput()) o.setStatus(o.states).setStates(true);

        Tenser<Tensor> nones = getOutput(getFunction());
        forEach(getOutput(), nones, (Tensor out, Tensor none) -> none.setGrad(out.grad));

        getFunction().forEach(Tensor::backward);
        clearGrad();

        for (Tensor o : getInput()) o.setStates(o.status).setStatus(false).backward();
    }

    public void reducer() {
        getFunction().forEach(Tensor::reducer);
        for (Tensor o : getInput()) o.reducer();
    }

    public Tenser<Tensor> getFunction() {
        if (Objects.nonNull(function)) return function;
        return function = compute();
    }

    public void clearGrad() {
        getOutput().forEach((Tensor out) -> out.setGradx(null));
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

}