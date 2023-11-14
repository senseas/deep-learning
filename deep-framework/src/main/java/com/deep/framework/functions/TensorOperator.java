package com.deep.framework.functions;

import com.deep.framework.lang.Tenser;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
        merge(this);
        for (Tensor o : getInput()) o.reducer();
    }

    public void clearGrad() {grad = null;}

    public Tenser<Tensor> getInput(int i) {
        return getInput()[i].getOutput();
    }

    public void concat(Tensor tensor) {
        if (List.of("Add", "Sum", "Mul", "Minusx").contains(tensor.getName())) {
            List<Tensor> input = new ArrayList<>();
            for (Tensor o : tensor.getInput()) {
                if (Objects.nonNull(o)) {
                    if (o.getName().equals(tensor.getName())) {
                        input.addAll(List.of(o.getInput()));
                    } else {
                        input.add(o);
                    }
                }
            }
            tensor.setInput(input.toArray(Tensor[]::new));
        }
    }

    public void merge(Tensor tensor) {
        if (!List.of("Add", "Sum", "Mul", "Minusx").contains(tensor.getName())) return;
        Map<String, List<Tensor>> map = Stream.of(tensor.getInput()).collect(Collectors.groupingBy(Tensor::getName));
        List<Tensor> list = new ArrayList<>();
        map.forEach((name, child) -> {
            if (child.size() < 2) {
                List<Tensor> c = child.stream().filter(a -> Objects.nonNull(a.getInput())).flatMap(a -> Stream.of(a.getInput())).toList();
                if (List.of("Add", "Sum", "Mul").contains(name) & c.size() == child.size()) {
                    list.addAll(c);
                } else {
                    list.addAll(child);
                }
            } else if (List.of("Add", "Sum", "Minusx").contains(name)) {
                Stream<Tensor> inputStream = child.stream().flatMap(a -> Stream.of(a.getInput()));
                Tensor input = child.get(0);
                input.setInput(inputStream.toArray(Tensor[]::new));
                list.add(input);
            } else if ("Mul".equals(name) && List.of("Add", "Sum", "Minusx").contains(tensor.getName())) {
                Stream<Tensor> inputStream = child.stream().flatMap(a -> Stream.of(a.getInput()));
                Map<Integer, List<Tensor>> mapx = inputStream.collect(Collectors.groupingBy(Tensor::getId));
                List<List<Tensor>> common = mapx.values().stream().filter(a -> a.size() == child.size()).toList();
                if (!common.isEmpty()) {
                    List<Tensor> commons = common.stream().flatMap(List::stream).toList();
                    Tensor[] addInput = child.stream().map(a -> {
                        Tensor[] input = Stream.of(a.getInput()).filter(b -> !commons.contains(b)).toArray(Tensor[]::new);
                        return input.length == 1 ? input[0] : mul(input).setId(a.getId());
                    }).toArray(Tensor[]::new);
                    Tensor input = mul(Stream.of(Stream.of(add(addInput)), common.stream().map(a -> a.get(0))).flatMap(a -> a).toArray(Tensor[]::new));
                    list.add(input);
                } else {
                    list.addAll(child);
                }
            } else if ("Div".equals(name)) {
                List<Tensor> iny = child.stream().map(a -> a.getInput()[1]).toList();
                Map<Integer, List<Tensor>> mapy = iny.stream().collect(Collectors.groupingBy(Tensor::getId));
                mapy.forEach((m, n) -> {
                    List<Tensor> input = child.stream().filter(b -> Stream.of(b.getInput()).map(Tensor::getId).toList().contains(n.get(0).getId())).toList();
                    if (n.size() > 2) {
                        List<Tensor> tensors = input.stream().map(c -> c.getInput()[0]).toList();
                        list.add(div(add(tensors.toArray(Tensor[]::new)), n.get(0)));
                    } else {
                        list.addAll(input);
                    }
                });
            } else {
                list.addAll(child);
            }
        });
        tensor.setInput(list.toArray(Tensor[]::new));
    }
}