package com.deep.framework.lang;

import jcuda.Pointer;
import jcuda.Sizeof;

import java.io.Serializable;
import java.util.Arrays;

import static com.deep.framework.cuda.Cuda.createDevicePointer;

public class Tenserx implements Serializable {

    public final Pointer deviceData;
    public final int[] shape, nexts;
    private final int offset, size;
    public int deviceId;

    public Tenserx(double[] data, int[] shape) {
        this.offset = 0;
        this.shape = shape;
        this.size = Shape.size(shape);
        this.deviceData = createDevicePointer(data, deviceId);
        this.nexts = next();
    }

    public Tenserx(double[] data, int[] shape, int offset, int deviceId) {
        this.offset = offset;
        this.shape = shape;
        this.size = Shape.size(shape);
        this.deviceData = createDevicePointer(data, deviceId).withByteOffset(offset * Sizeof.DOUBLE);
        this.nexts = next();
    }

    private Tenserx(Pointer deviceData, int[] shape, int offset) {
        this.offset = offset;
        this.shape = shape;
        this.size = Shape.size(shape);
        this.deviceData = deviceData.withByteOffset(offset * Sizeof.DOUBLE);
        this.nexts = next();
    }

    public Tenserx get(int... index) {
        return new Tenserx(this.deviceData, getNext(index), offset(index));
    }

    private int offset(int[] index) {
        int offset = this.offset;
        for (int i = 0; i < index.length; i++) offset += index[i] * nexts[i];
        return offset;
    }

    private int[] next() {
        int[] next = new int[shape.length];
        next[next.length - 1] = 1;
        for (int i = next.length - 1; 0 < i; i--) next[i - 1] = next[i] * shape[i];
        return next;
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

    public int shape(int i) { return shape[i]; }

    public int size() { return size; }

}