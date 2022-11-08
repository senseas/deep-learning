package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.graph.TensorConst;
import com.deep.framework.lang.Tenser;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.DoubleStream;

import static com.deep.framework.lang.ForEach.forEach;

public class TensorCorex implements Serializable {

    private List<double[]> list = new ArrayList<>();

    private void forward(Tensor tensor) {
        if (Objects.nonNull(tensor.getInput())) {
            for (Tensor o : tensor.getInput()) {
                forward(o);
            }
        } else if (!(tensor instanceof TensorConst)) {
            Object output = tensor.getOutput();
            if (Objects.nonNull(tensor.getValue())) {
                list.add(tensor.getValue());
            } else if (output instanceof Tenser) {
                Tenser<None> tenser = tensor.getOutput();
                list.add(tenser.stream().mapToDouble(None::getValue).toArray());
            } else {
                None tenser = tensor.getOutput();
                list.add(new double[]{tenser.getValue()});
            }
        }
    }

    public double[] getInput(Tensor tensor) {
        forEach(tensor.getFunction(), this::forward);
        return list.stream().flatMapToDouble(DoubleStream::of).toArray();
    }
}