package com.deep.framework.graph;

import com.deep.framework.framework.CudaContext;
import com.deep.framework.framework.CudaExecutor;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
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

    public int shape(int i) {return shape[i];}

    public CudaContext getContext() {
        if (Objects.nonNull(context)) return context;
        return context = CudaExecutor.New().createContext(this);
    }

    public Tensor setParams(Object... arr) {
        for (Object o : arr) {
            if (o instanceof List) {
                params.addAll((List) o);
            } else {
                params.add((double) o);
            }
        }
        return this;
    }

    private List<Double> params = new ArrayList<>();
    private String name = "Tensor::";
    protected int[] shape;
    private Tensor[] input;
    protected Object output, value, grad;
    protected transient Object function, reduce;
    private transient boolean gradre;
    private transient CudaContext context;
    private String grads = "1";

    public String toString() {
        return new StringBuilder("extern \"C\"")
            .append("__global__ void Sigmoid(double* inx , double* out)")
            .append("{")
            .append("  out[0] = ").append(BeanUtil.tmpl(grads, params))
            .append(";")
            .append("}").toString();
    }
}
