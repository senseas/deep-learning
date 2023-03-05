package com.deep.framework.framework;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;

import java.util.Objects;
import java.util.Optional;

public class TensorGeneCuda extends TensorGene {
    public String func = "", grad = "", funcCode = "", gradCode = "", name;

    public TensorGeneCuda() {}

    public TensorGeneCuda(String name) { this.name = name; }

    public void forward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            if (tensor.getFunction() instanceof Tenser) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();

                TensorGeneCuda cuda = new TensorGeneCuda(tensor.getName());
                cuda.forward(tenser.first());

                funcCode = funcCode.concat(cuda.getFuncCode());
                func = func.concat(tensor.getName().replace("Tensor::", "")).concat("<<<1," + tenser.size() + ">>>").concat("(in+" + inParams.size()).concat(",out+" + outParams.size()).concat(");");

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

                TensorGeneCuda cuda = new TensorGeneCuda(tensor.getName());
                cuda.backward(tenser.first());

                gradCode = gradCode.concat(cuda.getGradCode());
                grad = grad.concat(tensor.getName().replace("Tensor::", "Grad")).concat("<<<1," + tenser.size() + ">>>").concat("(in+" + inParams.size()).concat(",out+" + outParams.size()).concat(",outGrad+" + outGradParams.size()).concat(",inGrad+" + inGradParams.size()).concat(",innerGrad+" + innerGradParam.size()).concat(");");

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
        String names = Optional.ofNullable(name).orElse("compute").replace("Tensor::", "");
        return new StringBuilder("extern \"C\" __global__ void ")
        .append(names).append("(double* in, double* out){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("int M = ").append(outParams.size()).append(",").append("N = ").append(inParams.size()).append(";")
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
        String names = Optional.ofNullable(name).orElse("gradient").replace("Tensor::", "Grad");
        return new StringBuilder("extern \"C\" __global__ void ")
        .append(names).append("(double* in, double* out, double* outGrad, double* inGrad")
        .append(Objects.isNull(name) ? "){" : ",double* innerGrad){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append(Objects.nonNull(name) ? "" : "double innerGrad[").append(innerGradParam.size()).append("];")
        .append("int M = ").append(outParams.size()).append(",").append("N = ").append(inParams.size()).append(",").append("X = ").append(inGradParams.size()).append(",").append("Y = ").append(outGradParams.size()).append(";")
        .append(grad)
        .append("}")
        .append(gradCode)
        .toString();
    }

}