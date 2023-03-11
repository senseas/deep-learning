package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.driver.CUfunction;

import java.io.Serializable;
import java.util.Objects;

import static com.deep.framework.cuda.Cuda.createDeviceData;

public class CudaContext implements Serializable {

    private final Tensor tensor;
    private Pointer value, grad;
    private CUfunction function;

    public CudaContext(Tensor tensor) {
        this.tensor = tensor;
    }

    public Pointer getValue() {
        if (Objects.nonNull(value)) return value;
        return createDeviceData(tensor.getValue());
    }

    public Pointer getGrad() {
        if (Objects.nonNull(grad)) return grad;
        return createDeviceData(tensor.getGrad());
    }
}