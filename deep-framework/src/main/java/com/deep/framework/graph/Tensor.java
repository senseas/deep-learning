package com.deep.framework.graph;

import com.deep.framework.framework.CudaContext;
import com.deep.framework.framework.CudaExecutor;
import com.deep.framework.framework.TensorFlux;
import lombok.Data;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.deep.framework.lang.Shape.*;

@Data
public class Tensor implements Serializable {

    public Tensor(double value) {
        this.name = "Tensor";
        this.valuex = value;
        this.gradx = 0d;
        this.output = this;
        this.gradre = true;
    }

    public Tensor(double value, boolean gradre) {
        this.name = "Tensor";
        this.valuex = value;
        this.gradx = 0d;
        this.output = this;
        this.gradre = gradre;
    }

    public Tensor(int[] shape) {
        this.name = "Tensor";
        this.shape = shape;
        this.valuex = random(shape);
        this.gradx = zeros(shape);
        this.reducex = booleans(shape);
        this.output = fillNones(this);
        this.gradre = true;
    }

    public Tensor(String name, int[] shape) {
        this.name = "Tensor::".concat(name);
        this.shape = shape;
        this.valuex = random(shape);
        this.gradx = zeros(shape);
        this.reducex = booleans(shape);
        this.output = fillNones(this);
        this.gradre = true;
    }

    public Tensor(int[] shape, double value, boolean gradre) {
        this.name = "Tensor";
        this.shape = shape;
        this.valuex = values(shape, value);
        this.gradx = zeros(shape);
        this.reducex = booleans(shape);
        this.output = fillNones(this);
        this.gradre = gradre;
    }

    public Tensor(Tensor input) {
        this.name = "Tensor";
        this.output = input;
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public Tensor(Tensor tensor, int idx) {
        this.tensor = tensor;
        this.output = this;
        this.idx = idx;
    }

    public <M> Tensor(M m) {
        this.name = "Function";
        this.function = m;
        this.output = TensorFlux.getOutput(m);
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

    public double getValue() {
        if (Objects.isNull(tensor)) {
            return (double)this.valuex;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getValuex();
        } else {
            return ((double[]) tensor.getValuex())[idx];
        }
    }

    public void setValue(double value) {
        if (Objects.isNull(tensor)) {
            this.valuex = value;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setValuex(value);
        } else {
            ((double[]) tensor.getValuex())[idx] = value;
        }
    }

    public double getGrad() {
        if (Objects.isNull(tensor)) {
            return (double)this.gradx;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getGradx();
        } else {
            return ((double[]) tensor.getGradx())[idx];
        }
    }

    public void setGrad(double grad) {
        if (Objects.isNull(tensor)) {
            this.gradx =(double)this.gradx+ grad;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setGradx((double) tensor.getGradx() + grad);
        } else {
            ((double[]) tensor.getGradx())[idx] += grad;
        }
    }

    public boolean isReduce() {
        if (Objects.isNull(tensor)) {
            return (boolean) this.reducex;
        } else if (Objects.isNull(tensor.getShape())) {
            return (boolean) tensor.getReducex();
        } else {
            return ((boolean[]) tensor.getReducex())[idx];
        }
    }

    public void setReduce(boolean reduce) {
        if (Objects.isNull(tensor)) {
            this.reducex = reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReducex(reduce);
        } else {
            ((boolean[]) tensor.getReducex())[idx] = reduce;
        }
    }

    public void reset() {
        if (Objects.isNull(tensor)) {
            this.reducex = false;
            this.gradx = 0d;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReducex(false);
            tensor.setGradx(0d);
        } else {
            ((boolean[]) tensor.getReducex())[idx] = false;
            ((double[]) tensor.getGradx())[idx] = 0d;
        }
    }

    /*public Tensor setParams(Object... arr) {
        for (Object o : arr) {
            if (o instanceof List) {
                params.addAll((List) o);
            } else {
                params.add((double) o);
            }
        }
        return this;
    }*/

    private List<Double> params = new ArrayList<>();
    private String name = "Tensor::";
    protected int[] shape;
    private Tensor[] input;
    protected Object output, valuex = 0d, gradx = 0d;
    protected transient Object function, reducex;
    private transient boolean gradre;
    private transient CudaContext context;
    private int idx;
    private transient Tensor tensor;
    private String grads = "1";




}
