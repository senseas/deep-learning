package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;

import java.io.Serializable;
import java.util.Objects;

import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.runtime.JCuda.cudaMalloc;

public class CudaContext implements Serializable {

    private final Tensor tensor;
    private Pointer value, grad;

    public CudaContext(Tensor tensor) {
        this.tensor = tensor;
    }

    public Pointer getValue() {
        if (Objects.nonNull(value)) return value;
        double[] value = (double[]) tensor.getValue();
        Pointer pointer = new Pointer();
        cudaMalloc(pointer, value.length * Sizeof.DOUBLE);
        cublasSetVector(value.length, Sizeof.DOUBLE, Pointer.to(value), 1, pointer, 1);
        return pointer;
    }

    public Pointer getGrad() {
        if (Objects.nonNull(grad)) return grad;
        double[] value = (double[]) tensor.getGrad();
        Pointer pointer = new Pointer();
        cudaMalloc(pointer, value.length * Sizeof.DOUBLE);
        cublasSetVector(value.length, Sizeof.DOUBLE, Pointer.to(value), 1, pointer, 1);
        return pointer;
    }
}