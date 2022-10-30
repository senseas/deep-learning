package com.deep.framework.cuda;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import jcuda.driver.CUfunction;

import java.io.Serializable;
import java.util.Arrays;

import static com.deep.framework.cuda.Cuda.run;
import static com.deep.framework.lang.ForEach.forEach;

public class InputParallel implements Parallel {
    private CUfunction function;

    private double[] getInput(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).flatMapToDouble(a -> Arrays.stream(a.getInput()).mapToDouble(b -> ((None) b.getOutput()).getValue())).toArray();
    }

    public void compute(Tensor tensor) {
        Tensor[] tensors = tensor.getInput();
        int length = tensors[0].getOutParams().size();
        reset(tensor);

        double[] input = getInput(tensor);
        double[] output = new double[tensors.length * length];
        Cuda.run(function, new Dim(tensors.length), new Dim(1), input, output);

        tensor.setValuex(output);
        forEach(tensors.length, i -> tensors[i].getValue()[0] = output[i * length + length - 1]);
    }

    private void reset(Tensor tensor) {
        forEach(tensor.getOutput(), None::reset);
        Arrays.stream(tensor.getInput()).forEach(a -> Arrays.stream(a.getInput()).forEach(b -> ((None) b.getOutput()).reset()));
    }

    public void gradient(Tensor tensor) {
        Tensor[] tensors = tensor.getInput();
        int length = tensors[0].getInput().length;

        double[] input = getInput(tensor);
        double[] inGrad = new double[input.length];
        double[] outGrad = Arrays.stream(tensors).flatMapToDouble(a -> Arrays.stream(a.getGrad())).toArray();
        run(function, new Dim(tensors.length), new Dim(1), input, tensor.getValuex(), outGrad, inGrad);

        forEach(tensors.length, length, (i, l) -> tensors[i].getInput()[l].<None>getOutput().setGradx(inGrad[i * length + l]));
    }

}