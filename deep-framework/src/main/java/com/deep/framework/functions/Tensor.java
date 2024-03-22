package com.deep.framework.functions;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenser;
import lombok.Data;
import lombok.experimental.Accessors;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.fill;
import static com.deep.framework.lang.Shape.size;

@Data
@Accessors(chain = true)
public class Tensor implements Operator {

    public Tensor() {
        this.name = "None";
        this.data = "a" + id;
        this.output = new Tenser<>(this);
    }

    public Tensor(String value) {
        this.name = "None";
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
        this.output = new Tenser<>(this);
    }

    public Tensor(String name, int[] shape, Tensor... input) {
        this.name = name;
        this.shape = shape;
        this.input = input;
        this.output = Tensors();
    }

    public void forward() { grad = null; }

    public void backward() {}

    public void reducer() {}

    public String getVarId() {
        if (!forwed) return data;
        if (Objects.isNull(input)) return data;
        return "a" + id;
    }

    public String getGradId() { return "g" + id; }

    public void setGrad(Tensor grad) {
        this.grad = Objects.nonNull(this.grad) ? add(this.grad, grad) : grad;
    }

    public void setGradx(Tensor grad) { this.grad = grad; }

    public int shape(int i) { return shape[i]; }

    public Tenser<Tensor> Tensors() {
        return new Tenser<>(IntStream.range(0, size(shape)).mapToObj(i -> new Tensor()).toArray(Tensor[]::new), shape);
    }

    public static <E> E getOutput(Object a) {
        Object c = fill(a, Shape.shape(Object.class, a), b -> ((Tensor) b).getOutput());
        return (E) fill(c, Shape.shape(Tensor.class, c), b -> b);
    }

    public static <E> E zeroTensors(Object a) {
        return (E) fill(Shape.shape(Tensor.class, a), o -> new Tensor("0d"));
    }

    protected int[] shape;
    protected String data;
    protected Tensor grad;
    protected boolean status, states, forwed;

    private String name;
    private Tensor[] input;
    transient Tenser<Tensor> output;
    protected Tenser<Tensor> function;
    private int id = ID.getAndIncrement();
    private static AtomicInteger ID = new AtomicInteger();
    public static Map<String, Integer> map = new HashMap<>();

    @Override
    public String toString() {
        return "double a" + id + " = " + data + ";";
    }
}