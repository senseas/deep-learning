package com.deep.framework.cuda;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.Sizeof;

import static com.deep.framework.cuda.CublasConfig.handle;
import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class Matmul {

    //MK*KN
    public static void matmulForward(Tensor A, Tensor B, Tensor C) {
        // Allocate Copy the memory from the host to the device
        Pointer DA = createDevicePointer(A.getData());
        Pointer DB = createDevicePointer(B.getData());
        Pointer DC = createDevicePointer(C.getData());

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(1);
        // DC= [AD=NK * DB=KM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, DB, N, DA, K, beta, DC, N);
        // Copy the result from the device to the host
        cudaMemcpy(Pointer.to(C.getData()), DC, C.getData().length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        cudaFree(DA);
        cudaFree(DB);
        cudaFree(DC);
    }

    //MK*KN
    public static void matmulBackward(Tensor A, Tensor B, Tensor C) {
        // Allocate Copy the memory from the host to the device
        Pointer DA = createDevicePointer(A.getData());
        Pointer DB = createDevicePointer(B.getData());
        // Allocate Copy the memory from the host to the device
        Pointer GA = createDevicePointer(A.getGrad());
        Pointer GB = createDevicePointer(B.getGrad());
        Pointer GC = createDevicePointer(C.getGrad());

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(1);
        //GA= KM_T[DB=KN * GC_T=NM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, alpha, DB, N, GC, N, beta, GA, K);
        // Copy the result from the device to the host
        cudaMemcpy(Pointer.to(A.getGrad()), GA, A.getGrad().length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        //GB= NK_T[GC_T=NM * DA=MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, alpha, GC, N, DA, K, beta, GB, N);
        // Copy the result from the device to the host
        cudaMemcpy(Pointer.to(B.getGrad()), GB, B.getGrad().length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        cudaFree(DA);
        cudaFree(DB);
        cudaFree(GA);
        cudaFree(GB);
        cudaFree(GC);
    }

    //MK*NK
    public static void matmulTranbForward(Tensor A, Tensor B, Tensor C) {
        // Allocate Copy the memory from the host to the device
        Pointer DA = createDevicePointer(A.getData());
        Pointer DB = createDevicePointer(B.getData());
        Pointer DC = createDevicePointer(C.getData());

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(0);
        // DC= [AD=NK * DB=KM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, DB , K, DA, K, beta, DC, N);
        // Copy the result from the device to the host
        cudaMemcpy(Pointer.to(C.getData()), DC, C.getData().length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        cudaFree(DA);
        cudaFree(DB);
        cudaFree(DC);
    }

    //MK*NK
    public static void matmulTranbBackward(Tensor A, Tensor B, Tensor C) {
        // Allocate Copy the memory from the host to the device
        Pointer DA = createDevicePointer(A.getData());
        Pointer DB = createDevicePointer(B.getData());
        // Allocate Copy the memory from the host to the device
        Pointer GA = createDevicePointer(A.getGrad());
        Pointer GB = createDevicePointer(B.getGrad());
        Pointer GC = createDevicePointer(C.getGrad());

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(0);
        //GA= KM_T[DB=KN * GC_T=NM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, M, N, alpha, DB, K, GC, M, beta, GA, K);
        // Copy the result from the device to the host
        cudaMemcpy(Pointer.to(A.getGrad()), GA, A.getGrad().length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        //GB= NK_T[GC_T=NM * DA=MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, alpha, DA, K, GC, N, beta, GB, K);
        // Copy the result from the device to the host
        cudaMemcpy(Pointer.to(B.getGrad()), GB, B.getGrad().length * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);

        cudaFree(DA);
        cudaFree(DB);
        cudaFree(GA);
        cudaFree(GB);
        cudaFree(GC);
    }

}