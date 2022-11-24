package com.deep.framework.graph;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class TensorOparetor extends Tensor {

    public TensorOparetor(String name, Tensor... input) {
        super(name, input);
        List<List<Object>> list;
        if (1>1) {
            System.out.println(1);
        } else if (1>2) {
            System.out.println(2);
        } else {
            System.out.println(3);
        }

        if (Arrays.asList("Add").contains(name)) {
            Stream<Tensor> stream = Stream.of();
            for (Tensor o : input) {
                Stream<Tensor> children = o.getName().equals(getName()) ? Arrays.stream(o.getInput()) : Stream.of(o);
                stream = Stream.concat(stream, children);
            }
            setInput(stream.toArray(Tensor[]::new));
            int a = 1 + 2 * 3 * 4 + 5 + 7 * 8;
        }

        while (true){
            System.out.println("while");
        }

    }

    public None getInput(int i) {
        Tensor input = getInput()[i];
        return (None) input.getOutput();
    }

    public None getOutput() {
        return (None) output;
    }

    public void forward() {
        for (Tensor o : getInput()) TensorFlux.computer(o);
        TensorFlux.compute(this);
    }

    public void backward() {
        TensorFlux.gradient(this);
        for (Tensor o : getInput()) o.backward();
        for (Tensor o : getInput()) o.backward();
    }

    public void reduce() {
        for (Tensor o : getInput()) TensorFlux.reducer(o);
    }

}
