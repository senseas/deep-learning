package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class TensorOperator extends Tensor {

    public TensorOperator(String name, Tensor... input) {
        super(name, input);
        concat(this);
    }

    public String compute() {return data;}

    public void gradient(Tensor grad) {}

    public void forward() {
        if (status) return;
        for (Tensor o : getInput()) o.forward();
        data = compute();
        status = true;
    }

    public void backward() {
        gradient(grad);
        clearGrad();
        for (Tensor o : getInput()) o.backward();
    }

    public void reducer() {
        for (Tensor o : getInput()){
            o.reducer();
            concat(o);
        }
        merge(this);
    }

    public void clearGrad() {grad = null;}

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public void concat(Tensor tensor) {
        if (List.of("Add", "Mul", "Minusx").contains(tensor.getName())) {
            tensor.setInput(Stream.of(tensor.getInput()).flatMap((o) -> {
                if (o.getName().equals(tensor.getName())) {
                    return Stream.of(o.getInput());
                } else {
                    return Stream.of(o);
                }
            }).toArray(Tensor[]::new));
        }
    }

    public void merge(Tensor tensor) {
        if (List.of("Add", "Mul", "Minusx").contains(tensor.getName())) {
            Map<String, List<Tensor>> map = Stream.of(tensor.getInput()).collect(Collectors.groupingBy(Tensor::getName));
            List<Tensor> list = new ArrayList<>();
            map.forEach((name, child) -> {
                List<Tensor> tensors = child.stream().filter(a -> Objects.nonNull(a.getInput())).flatMap(a -> Stream.of(a.getInput())).toList();
                if (List.of("Add", "Mul").contains(name) && tensors.size() == 1) {
                    list.addAll(tensors);
                } else if ("Add".equals(tensor.getName()) && "Mul".equals(name) && child.size() > 1) {
                    Map<String, List<Tensor>> listMap = child.stream().flatMap(a -> Stream.of(a.getInput())).collect(Collectors.groupingBy(Tensor::getData));
                    List<List<Tensor>> common = listMap.values().stream().filter(a -> a.size() >= child.size()).toList();
                    if (!common.isEmpty()) {
                        List<Tensor> commons = common.stream().flatMap(List::stream).toList();
                        Tensor[] addInput = child.stream().map(a -> mul(Stream.of(a.getInput()).filter(b -> !commons.contains(b)).toArray(Tensor[]::new))).toArray(Tensor[]::new);
                        Tensor input = mul(Stream.of(Stream.of(add(addInput)), common.stream().map(a -> a.get(0))).flatMap(a -> a).toArray(Tensor[]::new));
                        list.add(input);
                    } else {
                        List<Tensor> childi = listMap.values().stream().min((a, b) -> b.size() - a.size()).get();
                        List<Tensor> childx = child.stream().filter(a -> Stream.of(a.getInput()).anyMatch(childi::contains)).toList();
                        if (childx.size() > 1) {
                            child.removeAll(childx);
                            Map<String, List<Tensor>> listMapx = childx.stream().flatMap(a -> Stream.of(a.getInput())).collect(Collectors.groupingBy(Tensor::getData));
                            List<List<Tensor>> commonx = listMapx.values().stream().filter(a -> a.size() >= childx.size()).toList();
                            if (!commonx.isEmpty()) {
                                List<Tensor> commons = commonx.stream().flatMap(List::stream).toList();
                                Tensor[] addInput = childx.stream().map(a -> mul(Stream.of(a.getInput()).filter(b -> !commons.contains(b)).toArray(Tensor[]::new))).toArray(Tensor[]::new);
                                Tensor input = mul(Stream.of(Stream.of(add(addInput)), commonx.stream().map(a -> a.get(0))).flatMap(a -> a).toArray(Tensor[]::new));
                                list.add(input);
                            } else {
                                list.addAll(childx);
                            }
                        }
                        list.addAll(child);
                    }
                } else {
                    list.addAll(child);
                }
            });

            tensor.setInput(list.toArray(Tensor[]::new));
            tensor.setStatus(false);
            tensor.forward();
        }
    }
}