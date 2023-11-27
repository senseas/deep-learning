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
                    Map<String, List<Tensor>> mapx = child.stream().flatMap(a -> {
                        Map<String, List<Tensor>> collect = Stream.of(a.getInput()).collect(Collectors.groupingBy(Tensor::getData));
                        return collect.values().stream().map(b -> b.get(0));
                    }).collect(Collectors.groupingBy(Tensor::getData));

                    List<List<Tensor>> list1 = mapx.values().stream().sorted((m, n) -> {
                        List<Tensor> parenta = child.stream().filter(a -> Stream.of(a.getInput()).anyMatch(m::contains)).toList();
                        List<Tensor> parentb = child.stream().filter(a -> Stream.of(a.getInput()).anyMatch(n::contains)).toList();

                        Map<String, List<Tensor>> mapa = parenta.stream().flatMap(a -> {
                            Map<String, List<Tensor>> collect = Stream.of(a.getInput()).collect(Collectors.groupingBy(Tensor::getData));
                            return collect.values().stream().map(b -> b.get(0));
                        }).collect(Collectors.groupingBy(Tensor::getData));
                        List<Tensor> commona = mapa.values().stream().filter(a -> a.size() >= parenta.size()).map(a -> a.get(0)).collect(Collectors.toList());

                        Map<String, List<Tensor>> mapb = parentb.stream().flatMap(a -> {
                            Map<String, List<Tensor>> collect = Stream.of(a.getInput()).collect(Collectors.groupingBy(Tensor::getData));
                            return collect.values().stream().map(b -> b.get(0));
                        }).collect(Collectors.groupingBy(Tensor::getData));

                        List<Tensor> commonb = mapb.values().stream().filter(a -> a.size() >= parentb.size()).map(a -> a.get(0)).collect(Collectors.toList());

                        return m.size() * commona.size() - n.size() * commonb.size();
                    }).toList();

                    list1.forEach((childx) -> {
                        List<Tensor> parent = child.stream().filter(a -> Stream.of(a.getInput()).anyMatch(childx::contains)).toList();
                        if (parent.size() <= 1) return;
                        Map<String, List<Tensor>> listMapx = parent.stream().flatMap(a -> {
                            Map<String, List<Tensor>> collect = Stream.of(a.getInput()).collect(Collectors.groupingBy(Tensor::getData));
                            return collect.values().stream().map(b -> b.get(0));
                        }).collect(Collectors.groupingBy(Tensor::getData));

                        List<Tensor> common = listMapx.values().stream().filter(a -> a.size() >= parent.size()).map(a -> a.get(0)).collect(Collectors.toList());
                        if(parent.stream().filter(a -> a.getInput().length == 1).count() == 0){
                            child.removeAll(parent);
                        }
                        if (!common.isEmpty() && parent.stream().filter(a -> a.getInput().length == 1).count() == 0) {
                            Tensor[] addInput = parent.stream().map(a -> {
                                Map<String, List<Tensor>> collect = Stream.of(a.getInput()).collect(Collectors.groupingBy(Tensor::getData));
                                common.stream().map(b -> collect.get(b.getData())).filter(Objects::nonNull).forEach(b -> b.remove(0));
                                return mul(collect.values().stream().flatMap(List::stream).toArray(Tensor[]::new));
                            }).toArray(Tensor[]::new);

                            common.add(add(addInput));
                            list.add(mul(common.toArray(Tensor[]::new)));
                        } else {
                            list.addAll(parent);
                        }
                    });
                    list.addAll(child);
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