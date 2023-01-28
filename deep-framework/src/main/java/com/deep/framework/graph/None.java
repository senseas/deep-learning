package com.deep.framework.graph;

import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

import static com.deep.framework.framework.TensorCore.*;

@Data
public class None implements Serializable {

    public None(Tensor tensor) {
        this.tensor = tensor;
        this.gradre = tensor.isGradre();
    }

    public None(Tensor tensor, int idx) {
        this.tensor = tensor;
        this.gradre = tensor.isGradre();
        this.idx = idx;
    }

    public None(double value) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
    }

    public None(double value, boolean gradre) {
        this.value = value;
        this.grad = 0d;
        this.reduce = false;
        this.gradre = gradre;
    }

    public double getValue() {
        if (Objects.isNull(tensor)) {
            return this.value;
        } else {
            return tensor.getValue()[idx];
        }
    }

    public void setValue(double value) {
        if (Objects.isNull(tensor)) {
            this.value = value;
        } else {
            tensor.getValue()[idx] = value;
        }
    }

    public double getGrad() {
        if (Objects.isNull(tensor)) {
            return this.grad;
        } else {
            return tensor.getGrad()[idx];
        }
    }

    public void setGrad(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad += grad;
        } else {
            tensor.getGrad()[idx] += grad;
        }
    }

    public void setGradx(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad = grad;
        } else {
            tensor.getGrad()[idx] = grad;
        }
    }

    public boolean isReduce() {
        if (Objects.isNull(tensor)) {
            return this.reduce;
        } else {
            return tensor.getReduce()[idx];
        }
    }

    public void setReduce(boolean reduce) {
        if (Objects.isNull(tensor)) {
            this.reduce = reduce;
        } else {
            tensor.getReduce()[idx] = reduce;
        }
    }

    public void reset() {
        if (Objects.isNull(tensor)) {
            this.reduce = false;
            this.grad = 0d;
        } else {
            tensor.getReduce()[idx] = false;
            tensor.getGrad()[idx] = 0d;
        }
    }

    public String getGradId() {
        if (BeanUtil.isNone(tensor)) return "inGrad[idx * N +" + idxInGrad.getAndIncrement() + "]";
        return "e" + id;
    }

    public String getValId() {
        if (tensor instanceof TensorConst) return "" + getValue();
        if (BeanUtil.isNone(tensor)) return "in[idx * N +" + idxIn.getAndIncrement() + "]";
        if (isForward) return "out[idx * M +" + idxOut.getAndIncrement() + "]";
        if (Objects.nonNull(valId)) return valId;
        return valId = "out[idx * M +" + idxOut.getAndIncrement() + "]";
    }

    public boolean isVal() {
        return !(tensor instanceof TensorConst);
    }

    private double value, grad;
    private transient boolean reduce, gradre;
    private transient int idx;
    private transient Tensor tensor;
    private int id = ID.getAndIncrement();
    private boolean root;
    private boolean isForward;
    private String valId;
    public transient static AtomicInteger ID = new AtomicInteger();
}