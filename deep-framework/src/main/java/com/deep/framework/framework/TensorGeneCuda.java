package com.deep.framework.framework;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;

import static com.deep.framework.framework.TensorFlux.getOutput;

public class TensorGeneCuda extends TensorGene {
    public String func = "", grad = "", funcCode = "", gradCode = "", name;

    public void forward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            if (tensor.getFunction() instanceof Tenser) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();

                TensorGeneCudaChild cuda = new TensorGeneCudaChild(this, tensor.getName());
                cuda.forward(tenser.first());
                ((Tenser<None>) tensor.getOutput()).first().setValId(((Tenser<None>) getOutput(tensor.getFunction())).first().getValId());

                funcCode = funcCode.concat(cuda.getFuncCode());
                func = func.concat(tensor.getName().replace("Tensor::", "")).concat("<<<1," + tenser.size() + ">>>").concat("(in + M,out + N);");

                TensorGeneContext context = new TensorGeneContext(this);
                tenser.forEach(context::forward);
            } else {
                Tensor func = (Tensor) tensor.getFunction();
                forward(func);
            }
        } else if (tensor instanceof TensorOperator) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            compute(tensor);
        }
    }

    public void backward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            if (tensor.getFunction() instanceof Tenser) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();
                ((Tenser<None>) getOutput(tensor.getFunction())).first().setValId((((Tenser<None>) tensor.getOutput()).first()).getValId());
                ((Tenser<None>) getOutput(tensor.getFunction())).first().setGradId((((Tenser<None>) tensor.getOutput()).first()).getGradId());

                TensorGeneCudaChild cuda = new TensorGeneCudaChild(this, tensor.getName());
                cuda.backward(tenser.first());

                gradCode = gradCode.concat(cuda.getGradCode());
                grad = grad.concat(tensor.getName().replace("Tensor::", "Grad")).concat("<<<1," + tenser.size() + ">>>").concat("(in + M, out + N, outGrad + Y, inGrad + X, innerGrad);");

                TensorGeneContext context = new TensorGeneContext(this);
                tenser.forEach(context::backward);
            } else {
                Tensor func = (Tensor) tensor.getFunction();
                backward(func);
            }
            for (Tensor o : tensor.getInput()) {
                backward(o);
            }
        } else if (tensor instanceof TensorOperator) {
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
    }

    public String getFuncCode() {
        return new StringBuilder("extern \"C\" __global__ ")
        .append("void compute(double* in, double* out){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("int M = idx * ").append(inParams.size()).append(",").append("N = idx * ").append(outParams.size()).append(";")
        .append(func)
        .append("}")
        .append(funcCode)
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
    }

    public String getGradCode() {
        return new StringBuilder("extern \"C\" __global__ ")
        .append("void gradient(double* in, double* out, double* outGrad, double* inGrad){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("double innerGrad[").append(innerGradParam.size()).append("];")
        .append("int M = idx * ").append(inParams.size()).append(",").append("N = idx * ").append(outParams.size()).append(",").append("X = idx * ").append(inGradParams.size()).append(",").append("Y = idx * ").append(outGradParams.size()).append(";")
        .append(grad)
        .append("}")
        .append(gradCode)
        .toString();
    }

}