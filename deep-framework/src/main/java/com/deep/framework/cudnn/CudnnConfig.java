package com.deep.framework.cudnn;

import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;
import jcuda.runtime.cudaStream_t;

import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.runtime.JCuda.cudaStreamCreate;

public class CudnnConfig {
    public static final cudnnHandle handle;

    static {
        JCudnn.setExceptionsEnabled(true);
        handle = new cudnnHandle();
        cudnnCreate(handle);
    }
}