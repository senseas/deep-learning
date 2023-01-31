package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorFunctor;

public class TensorGeneCuda extends TensorGene {
    public String func = "", grad = "", funcCode = "", gradCode = "";

    public void compute(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        output.setValId(null);
        output.setCore(this);
        setInputParam(tensor);

        functor.setInput(tensor.getInput());
        functor.setOut(output);

        func = func.concat(functor.compute());
        funcCode = getFuncCode();
    }

    private String getFuncCode() {
        return new StringBuilder()
        .append("extern \"C\" __global__ void compute(double* in, double* out){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("int M = ").append(outParams.size()).append(",")
        .append("N = ").append(inParams.size()).append(";")
        .append(func)
        .append("}")
        .toString();
    }

    public void gradient(Tensor tensor) {
        TensorFunctor functor = map.get(tensor.getName());
        None output = tensor.getOutput();
        output.setCore(this);
        setInputParam(tensor);

        functor.setInput(tensor.getInput());
        functor.setOut(output);

        grad = grad.concat(functor.gradient(""));
        gradCode = getGradCode();
    }

    private String getGradCode() {
        return new StringBuilder()
        .append("extern \"C\" __global__ void gradient(double* in, double* out, double* outGrad, double* inGrad){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("double ").append(String.join(",",innerGradParam)).append(";")
        .append("int M = ").append(outBackParams.size()).append(",")
        .append("N = ").append(inBackParams.size()).append(",")
        .append("X = ").append(inGradParams.size()).append(",")
        .append("Y = ").append(outGradParams.size()).append(";")
        .append(grad)
        .append("}")
        .toString();
    }
}