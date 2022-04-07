package com.deep.framework.lang;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;

import java.util.Objects;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaFree;

public class Cublas {

    private static cublasHandle handle;
    private static Cublas cublas;

    // Create a CUBLAS handle
    public Cublas() {
        handle = new cublasHandle();
        cublasCreate(handle);
    }

    public static Cublas New() {
        JCublas2.setExceptionsEnabled(true);
        if (Objects.nonNull(cublas)) return cublas;
        return cublas = new Cublas();
    }

    public void matmul(Tensor A, Tensor B, Tensor C) {
        double[] a = (double[]) A.getValue(), b = (double[]) B.getValue(), c = (double[]) C.getValue();

        // Allocate memory on the device
        // Copy the memory from the host to the device
        Pointer DA = A.getContext().getValue(), DB = B.getContext().getValue(), DC = C.getContext().getValue();

        // Execute sgemm
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        //MK*KN
        int M = A.getShape()[0],K = A.getShape()[1],N = B.getShape()[1];

        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, alpha, DA, K, DB, N, beta, DC, M);

        // Copy the result from the device to the host
        cublasGetVector(M * N, Sizeof.DOUBLE, DC, 1, Pointer.to(c), 1);
    }

    // Clean up
    public static void cudaFreex(CUdeviceptr d) {
        cudaFree(d);
        cublasDestroy(handle);
    }

}
