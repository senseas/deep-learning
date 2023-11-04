package com.deep.framework.cublas;

import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;

import static jcuda.jcublas.JCublas2.cublasCreate;

public class CublasConfig {
    public static cublasHandle handle;

    static {
        JCublas2.setExceptionsEnabled(true);
        handle = new cublasHandle();
        cublasCreate(handle);
    }
}