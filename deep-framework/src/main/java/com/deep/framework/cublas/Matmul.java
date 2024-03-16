package com.deep.framework.cublas;

import com.deep.framework.cuda.CudaContext;
import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;

import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

public class Matmul {

    //MK*KN
    public static void matmulForward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Pointer inputx_data = context.getDeviceData(inputx);
        Pointer inputy_data = context.getDeviceData(inputy);
        Pointer output_data = context.getDeviceData(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(1);
        // NM = [NK * KM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, inputy_data, N, inputx_data, K, beta, output_data, N);
        // Copy the result from the device to the host
        context.copyDataToHost(output);
        context.clear();
    }

    //MK*KN
    public static void matmulBackward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Pointer inputx_data = context.getDeviceData(inputx);
        Pointer inputy_data = context.getDeviceData(inputy);
        // Allocate Copy the memory from the host to the device
        Pointer inputx_grad = context.getDeviceGrad(inputx);
        Pointer inputy_grad = context.getDeviceGrad(inputy);
        Pointer output_grad = context.getDeviceGrad(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(1);
        // KM = [KN * NM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, alpha, inputy_data, N, output_grad, N, beta, inputx_grad, K);
        // Copy the result from the device to the host
        context.copyDataToHost(inputx);

        // NK = [NM * MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, alpha, output_grad, N, inputx_data, K, beta, inputy_grad, N);
        // Copy the result from the device to the host
        context.copyGradToHost(inputy);
        context.clear();
    }

    //MK*NK
    public static void matmulTranbForward(Tensor inputx, Tensor inputy, Tensor output, Tensor... alphas) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Pointer inputx_data = context.getDeviceData(inputx);
        Pointer inputy_data = context.getDeviceData(inputy);
        Pointer output_data = context.getDeviceData(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(0);
        // NM = [NK * KM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, inputy_data, K, inputx_data, K, beta, output_data, N);
        // Copy the result from the device to the host
        context.copyDataToHost(output);
        context.clear();
    }

    //MK*NK
    public static void matmulTranbBackward(Tensor inputx, Tensor inputy, Tensor output, Tensor... alphas) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Pointer inputx_data = context.getDeviceData(inputx);
        Pointer inputy_data = context.getDeviceData(inputy);
        // Allocate Copy the memory from the host to the device
        Pointer inputx_grad = context.getDeviceGrad(inputx);
        Pointer inputy_grad = context.getDeviceGrad(inputy);
        Pointer output_grad = context.getDeviceGrad(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(0);
        // KM = [KN * NM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, alpha, inputy_data, K, output_grad, N, beta, inputx_grad, K);
        // Copy the result from the device to the host
        context.copyGradToHost(inputx);

        // NK = [NM * MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, alpha, inputx_data, K, output_grad, N, beta, inputy_grad, K);
        // Copy the result from the device to the host
        context.copyGradToHost(inputy);
        context.clear();
    }

}