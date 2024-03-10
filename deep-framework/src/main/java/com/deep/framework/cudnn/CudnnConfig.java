package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnHandle;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.jcudnn.JCudnn.cudnnCreate;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

public class CudnnConfig {
    private static final Map<Integer, cudnnHandle> cudnnHandles = new HashMap<>();

    static {
        JCudnn.setExceptionsEnabled(true);
        int[] count = new int[1];
        cudaGetDeviceCount(count);
        IntStream.range(0, count[0]).forEach(CudnnConfig::create);
    }

    private static void create(int id) {
        cudaSetDevice(id);
        cudnnHandle cudnnHandle = new cudnnHandle();
        cudnnCreate(cudnnHandle);
        cudnnHandles.put(id, cudnnHandle);
    }

    public static cudnnHandle getCudnnHandle(Tensor tensor) {
        return getCudnnHandle(tensor.getDeviceId());
    }

    public static cudnnHandle getCudnnHandle(int id) {
        return cudnnHandles.get(id);
    }
}