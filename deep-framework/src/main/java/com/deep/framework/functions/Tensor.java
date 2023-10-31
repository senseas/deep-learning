package com.deep.framework.functions;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenser;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.fill;
import static com.deep.framework.lang.Shape.size;

@Data
@Accessors(chain = true)
public class Tensor implements Serializable {

    public Tensor(String value) {
        this.name = "None";
        this.data = value;
        this.output = new Tenser<>(this);
    }

    public Tensor(String name, String value) {
        this.name = name;
        this.data = value;
        this.output = new Tenser<>(this);
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.input = new Tensor[0];
        this.output = Tensors();
    }

    public Tensor(String name, Tensor... input) {
        this.name = name;
        this.input = input;
        output = new Tenser<>(this);
    }

    public Tensor(String name, int[] shape, Tensor... input) {
        this.name = name;
        this.shape = shape;
        this.input = input;
        this.output = Tensors();
    }

    public void forward() {}

    public void backward() {}

    public void reducer() {
        if (reduces) return;
        System.out.println("double ".concat(getGradId()).concat("=").concat(getGrad()));
        reduces = true;
    }

    public String getVarId() {return "a" + id;}

    public String getGradId() {return "g" + id;}

    public void setGrad(String grad) {
        this.grads.add(grad);
        this.grad = this.grad.equals("0d") ? grad : this.grad + "+" + grad;
    }

    public void setGradx(String grad) {
        this.grad = grad;
        this.grads.clear();
    }

    public int shape(int i) {return shape[i];}

    public Tenser<Tensor> Tensors() {
        return new Tenser<>(IntStream.range(0, size(shape)).mapToObj(i -> new Tensor(i + "")).toArray(Tensor[]::new), shape);
    }

    public static <E> E getOutput(Object a) {
        Object c = fill(a, Shape.shape(Object.class, a), b -> ((Tensor) b).getOutput());
        return (E) fill(c, Shape.shape(Tensor.class, c), b -> b);
    }

    public static <E> E zeroTensors(Object a) {
        return (E) fill(Shape.shape(Tensor.class, a), o -> new Tensor("0d"));
    }

    public String getGrad() {
        if (grads.isEmpty()) grads.add(grad);
        Map<Object, List<String>> map = grads.stream().collect(Collectors.groupingBy(a -> a));
        return map.values().stream().map(a -> {
            if (a.size() == 1) return a.get(0);
            String str = grad.startsWith("(") && grad.endsWith(")") ? a.get(0) : "*(" + a.get(0) + ")";
            return a.size() + str;
        }).collect(Collectors.joining("+")).concat(";");
    }

    protected int[] shape;
    protected String data = "", grad = "0d";
    protected boolean reduces;
    protected boolean status;

    private String name;
    private Tensor[] input;
    transient Tenser<Tensor> output;
    protected Tenser<Tensor> function;
    transient List<String> grads = new ArrayList<>();
    private int id = ID.getAndIncrement();
    private static AtomicInteger ID = new AtomicInteger();
}