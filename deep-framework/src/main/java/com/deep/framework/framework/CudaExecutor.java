package com.deep.framework.framework;

import com.deep.framework.graph.Tensor;
import lombok.Data;

import java.io.Serializable;
import java.util.Objects;

@Data
public class CudaExecutor<E> implements Serializable {
    private static CudaExecutor executor;

    public CudaContext createContext(Tensor tensor) {
        return new CudaContext(tensor);
    }

    public static CudaExecutor New() {
        if (Objects.nonNull(executor)) return executor;
        return executor = new CudaExecutor();
    }

}
