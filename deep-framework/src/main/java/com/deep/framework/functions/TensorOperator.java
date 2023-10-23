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
        data = "double ".concat(getVarId()).concat("=").concat(compute()).concat(";");
    }

    public void backward() {
        gradient(getGradId());
        for (Tensor o : getInput()) o.backward();
    }

    public void reducer() {
        System.out.println("double ".concat(getGradId()).concat("=").concat(this.grad));
        for (Tensor o : getInput()) o.reducer();
    }

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

}