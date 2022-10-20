package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.io.Serializable;
import java.util.Objects;

import static com.deep.framework.lang.cuda.Cuda.createDeviceData;

public class CudaContext implements Serializable {

    private final Tensor tensor;
    private Pointer value, grad;
    private CUfunction function;

    public CudaContext(Tensor tensor) {
        this.tensor = tensor;
    }

    public Pointer getValue() {
        if (Objects.nonNull(value)) return value;
        double[] value = (double[]) tensor.getValue();
        return createDeviceData(value);
    }

    public Pointer getGrad() {
        if (Objects.nonNull(grad)) return grad;
        double[] value = (double[]) tensor.getGrad();
        return createDeviceData(value);
    }
}