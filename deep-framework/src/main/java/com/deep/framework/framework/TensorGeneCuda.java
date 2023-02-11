package com.deep.framework.framework;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;

public class TensorGeneCuda extends TensorGene {
    public String func = "", grad = "", funcCode = "", gradCode = "";

    public void forward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            if (tensor.getFunction() instanceof Tenser) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
                func = func.concat("for (int i = 0; i <" + tenser.size() + " ; i++) {");
                index = "i";
                forward(tenser.first());
                setForParamsx(tenser.size());

                index = null;
                func = func.concat("}");
            } else {
                Tensor func = (Tensor) tensor.getFunction();
                forward(func);
            }
        } else if (tensor instanceof TensorOperator) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            setForwardParams();
            compute(tensor);
        }
    }

    public void backward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            if (tensor.getFunction() instanceof Tenser) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
                grad = grad.concat("for (int i = 0; i <" + tenser.size() + " ; i++) {");
                index = "i";
                backward(tenser.first());
                setBackParamsx(tenser.size());
                index = null;
                grad = grad.concat("}");
                gradCode = getGradCode();
            } else {
                Tensor func = (Tensor) tensor.getFunction();
                backward(func);
            }
            for (Tensor o : tensor.getInput()) {
                backward(o);
            }
        } else if (tensor instanceof TensorOperator) {
            setBackwardParams();
            gradient(tensor);
            for (Tensor o : tensor.getInput()) {
                backward(o);
            }
        }
    }

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
        .append("int M = ").append(outParamsx.stream().mapToInt(Integer::intValue).sum()).append(",")
        .append("N = ").append(inParamsx.stream().mapToInt(Integer::intValue).sum()).append(";")
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
        .append("int M = ").append(outBackParamsx.stream().mapToInt(Integer::intValue).sum()).append(",")
        .append("N = ").append(inBackParamsx.stream().mapToInt(Integer::intValue).sum()).append(",")
        .append("X = ").append(inGradParamsx.stream().mapToInt(Integer::intValue).sum()).append(",")
        .append("Y = ").append(outGradParamsx.stream().mapToInt(Integer::intValue).sum()).append(";")
        .append(grad)
        .append("}")
        .toString();
    }
}