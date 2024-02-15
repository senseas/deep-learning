package com.deep.framework.graph;

import com.deep.framework.lang.Tenser;
import com.deep.framework.optimizer.AdamOptimizer;
import jcuda.Pointer;
import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

import static com.deep.framework.cuda.Cuda.*;
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
        this.data = random(shape);
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(double[] data, int[] shape) {
        this.name = "None";
        this.shape = shape;
        this.data = data;
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(String name, int[] shape) {
        this.name = "None::".concat(name);
        this.shape = shape;
        this.data = random(shape);
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(int[] shape, double value) {
        this.name = "None";
        this.shape = shape;
        this.data = values(shape, value);
        this.grad = zeros(shape);
        this.reduce = true;
    }

    public Tensor(Tensor tensor, int idx) {
        this.idx = idx;
        this.tensor = tensor;
    }

    public Tensor(String name, Tensor... input) {
        this.name = this.name.concat(name);
        this.input = input;
    }

    public void forward() {
        if (Objects.nonNull(grad)) Arrays.fill(grad, 0d);
    }

    public void backward() { }

    public void reducer() {
        if (reduce) {
            createOptimizer();
            forEach(getOutput(), (Tensor none) -> optimizer.adam(none));
        }
    }

    public Tenser<Tensor> getOutput() {
        if (Objects.nonNull(output)) return output;
        if (Objects.isNull(shape)) return new Tenser<>(this);
        return output = Tensors(this);
    }

    public double data() {
        if (Objects.isNull(tensor)) {
            return this.data[idx];
        } else {
            return tensor.getData()[idx];
        }
    }

    public void data(double value) {
        if (Objects.isNull(tensor)) {
            this.data[idx] = value;
        } else {
            tensor.getData()[idx] = value;
        }
    }

    public double grad() {
        if (Objects.isNull(tensor)) {
            return this.grad[idx];
        } else {
            return tensor.getGrad()[idx];
        }
    }

    public void grad(double grad) {
        if (Objects.isNull(tensor)) {
            this.grad[idx] += grad;
        } else {
            tensor.getGrad()[idx] += grad;
        }
    }

    public void dataSynchronize() {
        if (Objects.isNull(deviceData)) return;
        copyDataDeviceToHost(data, deviceData);
    }

    public void gradSynchronize() {
        if (Objects.isNull(deviceGrad)) return;
        copyDataDeviceToHost(grad, deviceGrad);
    }

    public Pointer getDeviceData() {
        if (Objects.isNull(deviceData)) return deviceData = createDevicePointer(data);
        copyDataHostToDevice(data, deviceData);
        return deviceData;
    }

    public Pointer getDeviceGrad() {
        if (Objects.isNull(deviceGrad)) return deviceGrad = createDevicePointer(grad);
        copyDataHostToDevice(grad, deviceGrad);
        return deviceGrad;
    }

    public void createOptimizer() {
        if (Objects.nonNull(optimizer)) return;
        optimizer = new AdamOptimizer(shape);
    }

    public int shape(int i) { return shape[i]; }

    private String name = "";
    private Tensor[] input;
    protected int[] shape;
    protected double[] data, grad;

    transient private int idx;
    transient private Tensor tensor;
    transient private Pointer deviceData, deviceGrad;
    transient private AdamOptimizer optimizer;

    transient protected boolean status, statusx, reduce;
    transient protected Tenser<Tensor> output, function;
}