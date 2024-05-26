package com.deep.framework.graph;

import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenserx;

import java.util.Map;

public class Tensorx extends Tensor {

    public Tensorx(Tensor tensor, int[] shape, int start) {
        super(tensor, start);
        this.setStart(start);
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
        return getTensor().getData()[getIdx()];
    }

    public void data(double value) {
        getTensor().getData()[getIdx()] = value;
    }

    public double grad() {
        return getTensor().getGrad()[getIdx()];
    }

    public void grad(double grad) {
        getTensor().getGrad()[getIdx()] += grad;
    }

    public Map<Integer, Tenserx> getDeviceDataMap() {
        return getTensor().getDeviceDataMap();
    }

    public Map<Integer, Tenserx> getDeviceGradMap() {
        return getTensor().getDeviceGradMap();
    }
}