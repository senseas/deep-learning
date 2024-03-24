package com.deep.framework.cublas;

import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.runtime.JCuda.cudaGetDeviceCount;
import static jcuda.runtime.JCuda.cudaSetDevice;

public class CublasConfig {
    private static final Map<Integer, cublasHandle> cublasHandles = new HashMap<>();

    static {
        JCublas2.setExceptionsEnabled(true);
        int[] count = new int[1];
        cudaGetDeviceCount(count);
        IntStream.range(0, count[0]).forEach(CublasConfig::create);
    }

    private static void create(int id) {
        cudaSetDevice(id);
        cublasHandle cublasHandle = new cublasHandle();
        cublasCreate(cublasHandle);
        cublasHandles.put(id, cublasHandle);
    }

    public static cublasHandle getCublasHandle(int id) {
        return cublasHandles.get(id);
    }
}