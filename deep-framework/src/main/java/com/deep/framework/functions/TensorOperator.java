package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

public class TensorOperator extends Tensor {

    public TensorOperator(String name, Tensor... input) {
        super(name, input);
    }

    public String compute() {return data;}

    public void gradient(String grad) {}

    public void forward() {
        for (Tensor o : getInput()) o.forward();
        data = "double " + getVarId() + "=" + compute() + ";";
    }

    public void backward() {
        gradient(getGraId());
        for (Tensor o : getInput()) o.backward();
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

}