package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

import java.io.Serializable;
import java.util.Objects;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

public class CudaContext implements Serializable {

    private final Tensor tensor;
    private CUdeviceptr value, grad;

    public CudaContext(Tensor tensor) {
        this.tensor = tensor;
    }

    public CUdeviceptr getValue() {
        if (Objects.nonNull(value)) return value;
        double[] value = (double[]) tensor.getValue();
        CUdeviceptr deviceptr = new CUdeviceptr();
        cuMemAlloc(deviceptr, value.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceptr, Pointer.to(value), value.length * Sizeof.DOUBLE);
        return deviceptr;
    }

    public CUdeviceptr getGrad() {
        if (Objects.nonNull(grad)) return grad;
        double[] value = (double[]) tensor.getGrad();
        CUdeviceptr deviceptr = new CUdeviceptr();
        cuMemAlloc(deviceptr, value.length * Sizeof.DOUBLE);
        cuMemcpyHtoD(deviceptr, Pointer.to(value), value.length * Sizeof.DOUBLE);
        return deviceptr;
    }
}