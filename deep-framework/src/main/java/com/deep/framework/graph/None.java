package com.deep.framework.graph;

import com.deep.framework.framework.TensorGene;
import com.deep.framework.lang.util.BeanUtil;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;

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
        if (Objects.nonNull(core.index)) {
            if (BeanUtil.isNone(tensor)) {
                core.inGradParams.add(this);
                return "inGrad[idx * X +" + core.idxInGrad.getAndIncrement() + "+" + core.index + "]";
            }
            if (isOutGrad()) {
                core.outGradParams.add(this);
                return "outGrad[idx * Y +" + core.idxOutGrad.getAndIncrement() + "+" + core.index + "]";
            }
        } else {
            if (BeanUtil.isNone(tensor)) {
                core.inGradParams.add(this);
                return "inGrad[idx * X +" + core.idxInGrad.getAndIncrement() + "]";
            }
            if (isOutGrad()) {
                core.outGradParams.add(this);
                return "outGrad[idx * Y +" + core.idxOutGrad.getAndIncrement() + "]";
            }
        }
        String eid = "e" + id;
        core.innerGradParam.add(eid);
        return eid;
    }

    public String getValId() {
        if (tensor instanceof TensorConst) return "" + getValue();

        if (Objects.nonNull(core.index)) {
            if (core.isForward) {
                if (BeanUtil.isNone(tensor)) {
                    core.inParams.add(this);
                    return "in[idx * N +" + core.idxIn.getAndIncrement() + "+" + core.index + "]";
                }
                if (Objects.nonNull(valId)) return valId;
                core.outParams.add(this);
                return valId = "out[idx * M +" + core.idxOut.getAndIncrement() + "+" + core.index + "]";
            } else {
                if (BeanUtil.isNone(tensor)) {
                    core.inBackParams.add(this);
                    return "in[idx * N +" + core.idxIn.getAndIncrement() + "+" + core.index + "]";
                }
                core.outBackParams.add(this);
                return "out[idx * M +" + core.idxOut.getAndIncrement() + "+" + core.index + "]";
            }
        } else {
            if (core.isForward) {
                if (BeanUtil.isNone(tensor)) {
                    core.inParams.add(this);
                    return "in[idx * N +" + core.idxIn.getAndIncrement() + "]";
                }
                if (Objects.nonNull(valId)) return valId;
                core.outParams.add(this);
                return valId = "out[idx * M +" + core.idxOut.getAndIncrement() + "]";
            } else {
                if (BeanUtil.isNone(tensor)) {
                    core.inBackParams.add(this);
                    return "in[idx * N +" + core.idxIn.getAndIncrement() + "]";
                }
                core.outBackParams.add(this);
                return "out[idx * M +" + core.idxOut.getAndIncrement() + "]";
            }
        }
    }

    public boolean isVal() {
        return !(tensor instanceof TensorConst);
    }

    private double value, grad;
    private transient boolean reduce, gradre;
    private transient int idx;
    private transient Tensor tensor;
    private boolean isOutGrad;
    private TensorGene core;
    private String valId;
    private int id = ID.getAndIncrement();
    public transient static AtomicInteger ID = new AtomicInteger();
}