package com.deep.framework.functions;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenser;
import lombok.Data;

import java.io.Serializable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.deep.framework.lang.ForEach.fill;
import static com.deep.framework.lang.Shape.size;

@Data
public class Tensor implements Serializable {

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
        System.out.println("double ".concat(getGradId()).concat("=").concat(this.grad));
    }

    public String getVarId() {return "a" + id;}

    public String getGradId() {return "g" + id;}

    public void setGrad(String grad) {
        if (grad.equals("0d")) return;
        this.grad += "+" + grad;
    }

    public void setGradx(String grad) {
        this.grad = grad;
    }

    public int shape(int i) {return shape[i];}

    public Tenser<Tensor> Tensors() {
        Tensor[] tensors = IntStream.range(0, size(shape)).mapToObj(i -> new Tensor(i + "")).toArray(Tensor[]::new);
        return new Tenser<>(tensors, shape);
    }

    public static <E> E getOutput(Object a) {
        Object c = fill(a, Shape.shape(Object.class, a), b -> {
            Tensor o = (Tensor) b;
            return o.getOutput();
        });
        return (E) fill(c, Shape.shape(Tensor.class, c), b -> b);
    }

    public static <E> E zeroTensors(Object a) {
        return (E) fill(Shape.shape(Tensor.class, a), o -> new Tensor("0d"));
    }

    private int id = ID.getAndIncrement();
    public static AtomicInteger ID = new AtomicInteger();

    protected int[] shape;
    protected String data = "", grad = "0d";
    protected boolean[] reduce;
    protected boolean status;

    private String name;
    private Tensor[] input;
    transient Tenser<Tensor> output;
    protected Tenser<Tensor> function;
}