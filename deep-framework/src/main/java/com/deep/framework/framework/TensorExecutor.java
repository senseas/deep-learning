package com.deep.framework.framework;

import com.deep.framework.graph.None;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tensers;
import com.deep.framework.lang.function.Func2;
import lombok.Data;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;

import static com.deep.framework.lang.ForEach.forEach;

@Data
public class TensorExecutor<E> implements Serializable {
    public static double rate = 0.003;
    public static boolean status = true;
    public static AtomicInteger deep = new AtomicInteger();
    private Tensor tensor;
    private Tensor input, label;

    public TensorExecutor(Tensor tensor) {
        this.tensor = tensor;
    }

    public TensorExecutor(Tensor tensor, Tensor input, Tensor label) {
        this.tensor = tensor;
        this.input = input;
        this.label = label;
    }

    public void run(E input, E label) {
        setInput(input);
        setLabel(label);
        run();
    }

    public void run() {
        long ss = System.currentTimeMillis();
        tensor.forward();
        this.backward();
        tensor.reduce();
        status = false;
        //System.out.println(System.currentTimeMillis() - ss);
    }

    public void forward(E input, E label) {
        setInput(input);
        setLabel(label);
        tensor.forward();
    }

    public void backward() {
        forEach(tensor.getOutput(), (None none) -> none.setGrad(1d));
        tensor.backward();
    }

    public void reduce() {
        tensor.reduce();
    }

    public void setInput(Object o) {
        Func2<None, Double> func = (m, n) -> m.setValue(n);
        forEach(input.getOutput(), Tensers.tenser(o), func);
    }

    public void setLabel(Object o) {
        Func2<None, Double> func = (m, n) -> m.setValue(n);
        forEach(label.getOutput(), Tensers.tenser(o), func);
    }

}
