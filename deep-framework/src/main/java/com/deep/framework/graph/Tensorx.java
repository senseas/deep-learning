package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenserx;

import java.util.Map;

public class Tensorx extends Tensor {

    public Tensorx(Tensor tensor, int offset) {
        super(tensor, offset);
    }

    public Tensorx(Tensor tensor, int[] shape, int offset) {
        super(tensor, offset);
        this.setShape(shape);
        this.setSize(Shape.size(shape));
    }

    public double[] getData() {
        return getTensor().getData();
    }

    public double[] getGrad() {
        return getTensor().getGrad();
    }

    public double data() {
        return getTensor().getData()[getOffset()];
    }

    public void data(double value) {
        getTensor().getData()[getOffset()] = value;
    }

    public double grad() {
        return getTensor().getGrad()[getOffset()];
    }

    public void grad(double grad) {
        getTensor().getGrad()[getOffset()] += grad;
    }

    public boolean isReduce() {
        return getTensor().isReduce();
    }

    public Map<Integer, Tenserx> getDeviceDataMap() {
        return getTensor().getDeviceDataMap();
    }

    public Map<Integer, Tenserx> getDeviceGradMap() {
        return getTensor().getDeviceGradMap();
    }
}