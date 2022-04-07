package com.deep.framework.lang;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.cublasHandle;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaFree;

public class Cublas {

    private static cublasHandle handle;

    // Create a CUBLAS handle
    public Cublas() {
        handle = new cublasHandle();
        cublasCreate(handle);
    }

    public static void matmul(Tensor A, Tensor B, Tensor C) {
        double[] a = (double[]) A.getValue();
        double[] b = (double[]) B.getValue();
        double[] c = (double[]) C.getValue();

        // Allocate memory on the device
        // Copy the memory from the host to the device
        CUdeviceptr DA = A.getContext().getValue();
        CUdeviceptr DB = B.getContext().getValue();
        CUdeviceptr DC = B.getContext().getValue();

        // Execute sgemm
        Pointer alpha = Pointer.to(new double[]{1});
        Pointer beta = Pointer.to(new double[]{0});

        //MK*KN
        int M = A.getShape()[0];
        int K = A.getShape()[1];
        int N = B.getShape()[1];
        cublasDgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_T,
            M,
            N,
            K,
            alpha,
            DA,
            K,
            DB,
            N,
            beta,
            DC,
            N
        );

        // Copy the result from the device to the host
        cublasGetVector(M * N, Sizeof.DOUBLE, DC, 1, Pointer.to(c), 1);
    }

    // Clean up
    public static void cudaFreex(CUdeviceptr d) {
        cudaFree(d);
        cublasDestroy(handle);
    }

}
