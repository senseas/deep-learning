package com.deep.framework.graph;

import lombok.Data;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.Objects;

@Data
public class None implements Serializable {

    public None(Tensor tensor) {
        this.tensor = tensor;
    }

    public None(Tensor tensor, int idx) {
        this.tensor = tensor;
        this.idx = idx;
    }

    public None(double value) {
        this.value = value;
        this.grad = 0d;
        this.gradre = true;
        this.reduce = false;
    }

    public None(double value, boolean gradre) {
        this.value = value;
        this.grad = 0d;
        this.gradre = gradre;
        this.reduce = false;
    }

    public double getValue() {
        if (Objects.isNull(tensor)) {
            return this.value;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getValue();
        } else {
            return Array.getDouble(tensor.getValue(), idx);
        }
    }

    public void setValue(double value) {
        if (Objects.isNull(tensor)) {
            this.value = value;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setValue(value);
        } else {
            Array.setDouble(tensor.getValue(), idx, value);
        }
    }

    public double getGrad() {
        if (Objects.isNull(tensor)) {
            return this.grad;
        } else if (Objects.isNull(tensor.getShape())) {
            return (double) tensor.getGrad();
        } else {
            return Array.getDouble(tensor.getGrad(), idx);
        }
    }

    public void setGrad(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad += grad;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setGrad((double) tensor.getGrad() + grad);
        } else {
            Array.setDouble(tensor.getGrad(), idx, ((double[]) tensor.getGrad())[idx] + grad);
        }
    }

    public boolean isReduce() {
        if (Objects.isNull(tensor)) {
            return this.reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            return (boolean) tensor.getReduce();
        } else {
            return Array.getBoolean(tensor.getReduce(), idx);
        }
    }

    public void setReduce(boolean reduce) {
        if (Objects.isNull(tensor)) {
            this.reduce = reduce;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReduce(reduce);
        } else {
            Array.setBoolean(tensor.getReduce(), idx, reduce);
        }
    }

    public boolean getGradre() {
        if (Objects.isNull(tensor)) {
            return this.gradre;
        } else {
            return tensor.isGradre();
        }
    }

    public void setGradre(boolean gradre) {
        if (Objects.isNull(tensor)) {
            this.gradre = gradre;
        } else {
            tensor.setGradre(gradre);
        }
    }

    public void reset() {
        if (Objects.isNull(tensor)) {
            this.reduce = false;
            this.grad = 0d;
        } else if (Objects.isNull(tensor.getShape())) {
            tensor.setReduce(false);
            tensor.setGrad(0d);
        } else {
            Array.setBoolean(tensor.getReduce(), idx, false);
            Array.setDouble(tensor.getGrad(), idx, 0d);
        }
    }

    private int idx;
    private transient Tensor tensor;
    private double value, grad;
    private transient boolean gradre, reduce;
}
