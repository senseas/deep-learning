package com.deep.framework.creater;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;

public class ParamCreater extends Creater {

    public ParamCreater() { super(); }

    public ParamCreater(Creater creater) {
        super();
        inParams = creater.inParams;
        outParams = creater.outParams;
        inGradParams = creater.inGradParams;
        outGradParams = creater.outGradParams;
        innerGradParam = creater.innerGradParam;
    }

    public void compute(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        output.setValId(null);
        output.setCore(this);
        setInputParam(tensor);

        functor.setInput(tensor.getInput());
        functor.setOut(output);

        functor.compute();
    }

    public void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        output.setCore(this);
        setInputParam(tensor);

        functor.setInput(tensor.getInput());
        functor.setOut(output);

        functor.gradient("");
    }
}