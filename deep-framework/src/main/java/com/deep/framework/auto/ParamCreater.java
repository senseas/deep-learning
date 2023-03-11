package com.deep.framework.auto;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFunctor;

public class ParamCreater extends Creater {

    public ParamCreater() { super(); }

    public ParamCreater(Creater gene) {
        super();
        inParams = gene.inParams;
        outParams = gene.outParams;
        inGradParams = gene.inGradParams;
        outGradParams = gene.outGradParams;
        innerGradParam = gene.innerGradParam;
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