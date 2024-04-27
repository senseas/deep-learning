package com.deep.framework.lang;

import jcuda.Pointer;
import jcuda.Sizeof;

import java.io.Serializable;
import java.util.Arrays;

import static com.deep.framework.cuda.Cuda.createDevicePointer;

public class Tenserx implements Serializable {

    public final Pointer data, deviceData;
    public final int[] shape, nexts;
    private final int start, size;
    public int deviceId;

    public Tenserx(double[] data, int[] shape) {
        this.start = 0;
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = Pointer.to(data);
        this.deviceData = createDevicePointer(this.data, this.size, deviceId);
        this.nexts = next();
    }

    private Tenserx(Pointer data, Pointer deviceData, int[] shape, int start) {
        this.start = start;
        this.shape = shape;
        this.size = Shape.size(shape);
        this.data = data.withByteOffset(start * Sizeof.DOUBLE);
        this.deviceData = deviceData.withByteOffset(start * Sizeof.DOUBLE);
        this.nexts = next();
    }

    public Tenserx get(int... index) {
        return new Tenserx(this.data, this.deviceData, getNext(index), start(index));
    }

    private int start(int[] index) {
        int next = this.start, length = index.length - 1;
        for (int i = 0; i < length; i++) next += index[i] * nexts[i];
        return next + index[length] * nexts[length];
    }

    private int[] next() {
        int[] next = new int[shape.length];
        Arrays.fill(next, 1);
        for (int i = next.length - 1; 0 < i; i--) next[i - 1] = next[i] * shape[i];
        return next;
    }

    private int[] getNext(int[] index) {
        return Arrays.copyOfRange(this.shape, index.length, this.shape.length);
    }

    public int shape(int i) { return shape[i]; }

    public int size() { return size; }

}