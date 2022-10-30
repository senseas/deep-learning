package com.deep.framework.cuda;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenser;
import jcuda.driver.CUfunction;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.deep.framework.cuda.Cuda.run;

public class ReduceParallel implements Parallel {

    private CUfunction function;

    public double[] getInput(Tensor tensor) {
        return Arrays.stream(tensor.getInput()).flatMapToDouble(a -> Arrays.stream(a.getValue())).toArray();
    }

    public void compute(Tensor tensor) {
        Tenser<None> tenser = tensor.getOutput();
        Map<String, None> map = tenser.stream().collect(Collectors.toMap(a -> a.getValId().trim(), a -> a));
        int length = tensor.getOutParams().size();

        double[] input = getInput(tensor);
        double[] output = new double[length];

        Cuda.run(function, new Dim(1), new Dim(1), input, output);
        tensor.setValues(output);
        tensor.setValue(IntStream.range(0, length).filter(i -> Objects.nonNull(map.get(tensor.getOutParams().get(i)))).mapToDouble(i -> output[i]).toArray());
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