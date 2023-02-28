package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFunctor;

public class TensorGeneContext extends TensorGene {

    public TensorGeneContext() { super(); }

    public TensorGeneContext(TensorGene gene) {
        super();
        inParams = gene.inParams;
        outParams = gene.outParams;
        inGradParams = gene.inGradParams;
        outGradParams = gene.outGradParams;
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