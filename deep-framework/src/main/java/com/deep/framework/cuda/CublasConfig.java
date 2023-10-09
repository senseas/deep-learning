package com.deep.framework.cuda;

import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcudnn.JCudnn.cudnnCreate;

public class CublasConfig {
    public static cublasHandle handle;

    static {
        JCublas2.setExceptionsEnabled(true);
        handle = new cublasHandle();
        cublasCreate(handle);
    }
}