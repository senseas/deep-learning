package com.deep.framework.cublas;

import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

public class CublasConfig {
    private static final Map<Long, cublasHandle> cublasHandles = new HashMap<>();

    static {
        JCublas2.setExceptionsEnabled(true);
        int[] count = new int[1];
        cudaGetDeviceCount(count);
    }

    private static void create(int id, long threadId) {
        cudaSetDevice(id);
        cublasHandle cublasHandle = new cublasHandle();
        cublasCreate(cublasHandle);
        cublasHandles.put(threadId, cublasHandle);
    }

    public static cublasHandle getCublasHandle(int id) {
        long threadId = Thread.currentThread().getId();
        cublasHandle handle = cublasHandles.get(threadId);
        if (Objects.nonNull(handle)) return handle;
        create(id, threadId);
        return cublasHandles.get(threadId);
    }
}