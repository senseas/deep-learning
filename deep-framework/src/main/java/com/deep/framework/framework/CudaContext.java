package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;

@Data
public class CudaContext implements Serializable {

    private final Tensor tensor;
    private Pointer value, grad;
    private CUfunction function;

    public CudaContext(Tensor tensor, CUfunction function) {
        this.tensor = tensor;
        this.function = function;
    }

    public Pointer getValue() {
        if (Objects.nonNull(value)) return value;
        double[] value = (double[]) tensor.getValue();
        return CudaExecutor.New().createDeviceData(value);
    }

    public Pointer getGrad() {
        if (Objects.nonNull(grad)) return grad;
        double[] value = (double[]) tensor.getGrad();
        return CudaExecutor.New().createDeviceData(value);
    }
}