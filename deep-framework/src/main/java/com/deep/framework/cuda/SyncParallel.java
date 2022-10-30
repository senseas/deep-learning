package com.deep.framework.cuda;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import jcuda.driver.CUfunction;

import java.util.Arrays;
import java.util.stream.IntStream;

import static com.deep.framework.cuda.Cuda.run;

public class SyncParallel implements Parallel {

    private CUfunction function;

    public double[] getInput(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).flatMapToDouble(a -> Arrays.stream(a.getValue())).toArray();
    }

    public void compute(Tensor tensor) {
        int length = tensor.getOutParams().size();

        double[] input = getInput(tensor);
        double[] output = new double[length];
        run(function, input, output);

        None none = tensor.getOutput();
        tensor.setValues(output);
        none.setValue(output[length - 1]);
    }

    public void gradient(Tensor tensor) {
        double[] input = getInput(tensor);
        double[] inGrad = new double[input.length];
        run(function, input, tensor.getValues(), tensor.getGrad(), inGrad);

        int length = tensor.getInput().length, l = input.length / length;
        IntStream.range(0, length).forEach(i -> {
            int from = i * l;
            Tensor in = tensor.getInput()[i];
            in.setGrad(Arrays.copyOfRange(inGrad, from, from + l));
        });
    }

}