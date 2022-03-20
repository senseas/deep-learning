package com.deep.framework.graph;

import com.deep.framework.framework.TensorContext;
import com.deep.framework.framework.TensorGpuExecutor;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

@Data
public class Tensor implements Serializable {

    public Tensor(double value) {
        this.name = "None";
        this.value = value;
        this.grad = 0d;
        this.output = new None(this);
        this.gradre = true;
    }

    public Tensor(double value, boolean gradre) {
        this.name = "None";
        this.value = value;
        this.grad = 0d;
        this.output = new None(this);
        this.gradre = gradre;
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.value = random(shape);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
        this.output = fillNones(this);
        this.gradre = true;
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.shape = shape;
        this.value = random(shape);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
        this.output = fillNones(this);
        this.gradre = true;
    }

    public Tensor(int[] shape, double value, boolean gradre) {
        this.name = "None";
        this.shape = shape;
        this.value = values(shape, value);
        this.grad = zeros(shape);
        this.reduce = booleans(shape);
        this.output = fillNones(this);
        this.gradre = gradre;
    }

    public Tensor(None input) {
        this.name = "None";
        this.output = input;
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public <M> Tensor(M m) {
        this.name = "Function";
        this.function = m;
    }

    public <M> M compute() { return null; }

    public void gradient() { }

    public void forward() { }

    public void backward() { }

    public void reduce() { }

    public <M> M getOutput() { return (M) output; }

    public TensorContext getContext() {
        if (Objects.nonNull(context)) return context;
        return context = TensorGpuExecutor.New().createContext(this);
    }

    private String name = "Tensor::";
    protected int[] shape;
    private Tensor[] input;
    protected Object output, value, grad;
    protected transient Object function, reduce;
    private transient boolean gradre;
    private transient TensorContext context;
}
