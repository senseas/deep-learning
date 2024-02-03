package com.deep.framework.cublas;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.runtime.cudaStream_t;

import static com.deep.framework.cublas.CublasConfig.handle;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.JCublas2.cublasSetStream;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaStreamCreate;
import static jcuda.runtime.JCuda.cudaStreamDestroy;

public class Matmul {

    //MK*KN
    public static void matmulForward(Tensor A, Tensor B, Tensor C) {
        // Allocate Copy the memory from the host to the device
        Pointer DA = A.getDeviceData();
        Pointer DB = B.getDeviceData();
        Pointer DC = C.getDeviceData();

        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cublasSetStream(handle, stream);

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(1);
        // DC= [AD=NK * DB=KM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, DB, N, DA, K, beta, DC, N);
        // Copy the result from the device to the host
        C.dataSynchronize();

        cudaStreamDestroy(stream);
    }

    //MK*KN
    public static void matmulBackward(Tensor A, Tensor B, Tensor C) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cublasSetStream(handle, stream);

        // Allocate Copy the memory from the host to the device
        Pointer DA = A.getDeviceData();
        Pointer DB = B.getDeviceData();
        // Allocate Copy the memory from the host to the device
        Pointer GA = A.getDeviceGrad();
        Pointer GB = B.getDeviceGrad();
        Pointer GC = C.getDeviceGrad();

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(1);
        //GA= KM_T[DB=KN * GC_T=NM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, alpha, DB, N, GC, N, beta, GA, K);
        // Copy the result from the device to the host
        A.gradSynchronize();

        //GB= NK_T[GC_T=NM * DA=MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, alpha, GC, N, DA, K, beta, GB, N);
        // Copy the result from the device to the host
        B.gradSynchronize();

        cudaStreamDestroy(stream);
    }

    //MK*NK
    public static void matmulTranbForward(Tensor A, Tensor B, Tensor C, Tensor... alphas) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cublasSetStream(handle, stream);

        // Allocate Copy the memory from the host to the device
        Pointer DA = A.getDeviceData();
        Pointer DB = B.getDeviceData();
        Pointer DC = C.getDeviceData();

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(0);
        // DC= [AD=NK * DB=KM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, DB, K, DA, K, beta, DC, N);
        // Copy the result from the device to the host
        C.dataSynchronize();

        cudaStreamDestroy(stream);
    }

    //MK*NK
    public static void matmulTranbBackward(Tensor A, Tensor B, Tensor C, Tensor... alphas) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cublasSetStream(handle, stream);

        // Allocate Copy the memory from the host to the device
        Pointer DA = A.getDeviceData();
        Pointer DB = B.getDeviceData();
        // Allocate Copy the memory from the host to the device
        Pointer GA = A.getDeviceGrad();
        Pointer GB = B.getDeviceGrad();
        Pointer GC = C.getDeviceGrad();

        //alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = A.shape(0), K = A.shape(1), N = B.shape(0);
        //GA= KM_T[DB=KN * GC_T=NM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, M, N, alpha, DB, K, GC, M, beta, GA, K);
        // Copy the result from the device to the host
        A.gradSynchronize();

        //GB= NK_T[GC_T=NM * DA=MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, N, M, alpha, DA, K, GC, N, beta, GB, K);
        // Copy the result from the device to the host
        B.gradSynchronize();

        cudaStreamDestroy(stream);
    }

}