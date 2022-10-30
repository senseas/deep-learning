package com.deep.framework.cuda;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import jcuda.driver.CUfunction;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.IntStream;

import static com.deep.framework.cuda.Cuda.run;

public class FunctionParallel implements Parallel {

    private CUfunction function;

    public double[] getInput(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).flatMapToDouble(a -> Arrays.stream(a.getValue())).toArray();
    }

    public void compute(Tensor tensor) {
        Tenser<None> tenser = tensor.getOutput();
        int size = tenser.size(), length = tensor.getOutParams().size();

        double[] input = getInput(tensor);
        double[] output = new double[size * length];
        Cuda.run(function, new Dim(size), new Dim(1), input, output);

        tensor.setValues(output);
        tensor.setValue(IntStream.range(0, size).mapToDouble(i -> output[i * length + length - 1]).toArray());
    }

    public void gradient(Tensor tensor) {
        Tenser<None> tenser = tensor.getOutput();

        double[] input = getInput(tensor);
        double[] inGrad = new double[input.length];
        run(function, new Dim(tenser.size()), new Dim(1), input, tensor.getValues(), tensor.getGrad(), inGrad);

        int length = tensor.getInput().length, l = input.length / length;
        IntStream.range(0, length).forEach(i -> {
            int from = i * l;
            Tensor in = tensor.getInput()[i];
            in.setGrad(Arrays.copyOfRange(inGrad, from, from + l));
        });
    }
}