package com.deep.framework.cudnn;

import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

public class CudnnConfig {
    private static final Map<Long, cudnnHandle> cudnnHandles = new HashMap<>();

    static {
        JCudnn.setExceptionsEnabled(true);
        int[] count = new int[1];
        cudaGetDeviceCount(count);
    }

    private static void create(int id, long threadId) {
        cudaSetDevice(id);
        cudnnHandle cudnnHandle = new cudnnHandle();
        cudnnCreate(cudnnHandle);
        cudnnHandles.put(threadId, cudnnHandle);
    }

    public static cudnnHandle getCudnnHandle(int id) {
        long threadId = Thread.currentThread().getId();
        cudnnHandle handle = cudnnHandles.get(threadId);
        if (Objects.nonNull(handle)) return handle;
        create(id, threadId);
        return cudnnHandles.get(threadId);
    }
}