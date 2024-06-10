package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenser;
import com.deep.framework.lang.Tenserx;
import com.deep.framework.optimizer.AdamOptimizer;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static com.deep.framework.lang.Shape.Tensors;
import static com.deep.framework.lang.Shape.*;

@Data
@Accessors(chain = true)
public class Tensor implements Serializable {

    public Tensor(double value) {
        this.name = "None";
        this.data = new double[]{value};
        this.grad = new double[]{0d};
        this.reduce = true;
    }

    public Tensor(int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = random(shape);
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(double[] data, int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = data;
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = random(shape);
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(int[] shape, double value) {
        this.name = "None";
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = values(shape, value);
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(Tensor tensor) {
        this.shape = tensor.getShape();
        this.size = Shape.size(shape);
        this.data = tensor.getData();
        this.grad = tensor.getGrad();
    }

    public Tensor(Tensor tensor, int offset) {
        this.offset = offset;
        this.tensor = tensor;
    }

    public Tensor(String name, int[] shape, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
        if (Objects.isNull(shape)) return;
        this.shape = shape;
        this.size = Shape.size(shape);
    }

    public void forward() {
        if (Objects.nonNull(grad)) Arrays.fill(grad, 0d);
    }

    public void backward() { }

    public void reducer() {
        if (reduce) {
            createOptimizer();
            forEach(this, none -> optimizer.adam(none));
        }
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        if (Objects.isNull(shape)) return new Tenser<>(this);
        return output = Tensors(this);
    }

    public double data() {
        if (Objects.isNull(tensor)) {
            return this.data[offset];
        } else {
            return tensor.getData()[offset];
        }
    }

    public void data(double value) {
        if (Objects.isNull(tensor)) {
            this.data[offset] = value;
        } else {
            tensor.getData()[offset] = value;
        }
    }

    public double grad() {
        if (Objects.isNull(tensor)) {
            return this.grad[offset];
        } else {
            return tensor.getGrad()[offset];
        }
    }

    public void grad(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad[offset] += grad;
        } else {
            tensor.getGrad()[offset] += grad;
        }
    }

    public void setStatus(boolean status) {
        if (Objects.isNull(tensor)) {
            this.status = status;
        } else {
            tensor.setStatus(status);
        }
    }

    public boolean isStatus() {
        if (Objects.isNull(tensor)) {
            return this.status;
        } else if (tensor.isReduce()) {
            return true;
        } else {
            return tensor.isStatus();
        }
    }

    public void createOptimizer() {
        if (Objects.nonNull(optimizer)) return;
        optimizer = new AdamOptimizer(shape);
    }

    public Map<Integer, Tenserx> getDeviceDataMap() {
        if (Objects.nonNull(deviceDataMap)) return deviceDataMap;
        return deviceDataMap = new HashMap<>();
    }

    public Map<Integer, Tenserx> getDeviceGradMap() {
        if (Objects.nonNull(deviceGradMap)) return deviceGradMap;
        return deviceGradMap = new HashMap<>();
    }

    public int shape(int i) {return shape[i];}

    public int size() {return size;}

    public Tensor get(int... index) {
        if (shape[0] == 1) index[0] = 0;
        if (shape.length == 1) return new Tensorx(this, new int[0], offset(index));
        int[] shapeNext = Arrays.copyOfRange(this.shape, index.length, this.shape.length);
        return new Tensorx(this, shapeNext, offset(index));
    }

    public Tensor getx(int index) {
        return new Tensorx(this, new int[]{1}, this.offset + index);
    }

    private int offset(int[] index) {
        int[] nexts = getNext();
        int next = this.offset, length = index.length - 1;
        for (int i = 0; i < length; i++) next += index[i] * nexts[i];
        return next + index[length] * nexts[length];
    }

    public int[] getNext(int... shape) {
        shape = shape.length > 0 ? shape : this.shape;
        int[] next = new int[shape.length];
        Arrays.fill(next, 1);
        for (int i = next.length - 1; 0 < i; i--) next[i - 1] = next[i] * shape[i];
        return next;
    }

    private String name = "";
    private Tensor[] input;
    private Tensor tensor;
    private int offset, size = 1;

    protected int[] shape = new int[]{1};
    protected double[] data, grad;
    protected boolean reduce, status;
    protected Tenser<Tensor> output, function;

    transient private AdamOptimizer optimizer;

    transient private int deviceId;
    transient private Map<Integer, Tenserx> deviceDataMap;
    transient private Map<Integer, Tenserx> deviceGradMap;
}