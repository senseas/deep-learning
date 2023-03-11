package com.deep.framework.creater;

import com.deep.framework.graph.*;
import com.deep.framework.lang.Tenser;

public class CudaSubCreater extends Creater {
    public String func = "", grad = "", funcCode = "", gradCode = "", name;

    public CudaSubCreater(Creater creater, String name) {
        inxParams.addAll(creater.inxParams);
        outxParams.addAll(creater.outxParams);

        inParams.addAll(creater.inParams);
        outParams.addAll(creater.outParams);
        inGradParams.addAll(creater.inGradParams);
        outGradParams.addAll(creater.outGradParams);
        innerGradParam.addAll(creater.innerGradParam);
        this.name = name;
    }

    public void forward(Tensor tensor) {
        if (tensor instanceof TensorFunction) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
            if (tensor.getFunction() instanceof Tenser) {
                Tenser<Tensor> tenser = (Tenser<Tensor>) tensor.getFunction();

                CudaSubCreater cuda = new CudaSubCreater(this, tensor.getName());
                cuda.forward(tenser.first());

                funcCode = funcCode.concat(cuda.getFuncCode());
                func = func.concat(tensor.getName().replace("Tensor::", "")).concat("<<<1," + tenser.size() + ">>>").concat("(in + M, out + N);");

                ParamCreater context = new ParamCreater(this);
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

                CudaSubCreater cuda = new CudaSubCreater(this, tensor.getName());
                cuda.backward(tenser.first());

                gradCode = gradCode.concat(cuda.getGradCode());
                grad = grad.concat(tensor.getName().replace("Tensor::", "Grad")).concat("<<<1," + tenser.size() + ">>>").concat("(in + M, out + N, outGrad + Y, inGrad + X, innerGrad);");

                ParamCreater context = new ParamCreater(this);
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
        String names = name.replace("Tensor::", "");
        return new StringBuilder("extern \"C\" __global__ void ")
        .append(names).append("(double* in, double* out){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("int M = idx, N = idx;")
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
        String names = name.replace("Tensor::", "Grad");
        return new StringBuilder("extern \"C\" __global__ void ")
        .append(names).append("(double* in, double* out, double* outGrad, double* inGrad, double* innerGrad){")
        .append("int idx = blockDim.x * blockIdx.x + threadIdx.x;")
        .append("int M = idx, N = idx, X = idx, Y = idx;")
        .append(grad)
        .append("}")
        .append(gradCode)
        .toString();
    }

}