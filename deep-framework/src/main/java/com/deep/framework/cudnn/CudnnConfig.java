package com.deep.framework.cudnn;

import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;

import static jcuda.jcudnn.JCudnn.cudnnCreate;

public class CudnnConfig {
    public static final cudnnHandle handle;

    static {
        JCudnn.setExceptionsEnabled(true);
        handle = new cudnnHandle();
        cudnnCreate(handle);
    }
}