package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;

import static com.deep.framework.cuda.CublasConfig.handle;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

public class Matmul {

    //MK*KN
    public static void matmulForward(Tensor A, Tensor B, Tensor C) {
        // Allocate Copy the memory from the host to the device
        Pointer DA = A.getContext().getValue(), DB = B.getContext().getValue(), DC = C.getContext().getValue();
        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.getShape()[0], K = A.getShape()[1], N = B.getShape()[1];
        // DC= [AD=NK * DB=KM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, DB, N, DA, K, beta, DC, N);
        // Copy the result from the device to the host
        cublasGetVector(M * N, Sizeof.DOUBLE, DC, 1, Pointer.to(C.getData()), 1);
    }

    //MK*KN
    public static void matmulBackward(Tensor A, Tensor B, Tensor C) {
        // Allocate Copy the memory from the host to the device
        Pointer DA = A.getContext().getValue(), DB = B.getContext().getValue();
        // Allocate Copy the memory from the host to the device
        Pointer GA = A.getContext().getGrad(), GB = B.getContext().getGrad(), GC = C.getContext().getGrad();
        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.getShape()[0], K = A.getShape()[1], N = B.getShape()[1];
        //GA= KM_T[DB=KN * GC_T=NM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, alpha, DB, N, GC, N, beta, GA, K);
        // Copy the result from the device to the host
        cublasGetVector(M * K, Sizeof.DOUBLE, GA, 1, Pointer.to(A.getGrad()), 1);

        //GB= NK_T[GC_T=NM * DA=MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, alpha, GC, N, DA, K, beta, GB, N);
        // Copy the result from the device to the host
        cublasGetVector(K * N, Sizeof.DOUBLE, GB, 1, Pointer.to(B.getGrad()), 1);
    }

    public void matmulForward(int transa, int transb, int M, int N, int K, Pointer A, Pointer B, Pointer C) {
        int lda = transa == CUBLAS_OP_N ? K : M, ldb = transb == CUBLAS_OP_N ? N : K, ldc = N;
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cublasDgemm(handle, transb, transa, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc);
    }

}