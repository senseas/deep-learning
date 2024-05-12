package com.deep.framework.cublas;

import com.deep.framework.cuda.CudaContext;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Tenserx;
import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;

import static com.deep.framework.lang.ForEach.forEach;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

public class Matmul {

    //MK*KN
    public static void matmulForward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Tenserx inputx_data = context.getDeviceData(inputx);
        Tenserx inputy_data = context.getDeviceData(inputy);
        Tenserx output_data = context.getDeviceData(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(1), K = inputx.shape(2), N = inputy.shape(2);
        // NM = [NK * KM]
        forEach(inputx.shape(0), i -> cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, inputy_data.get(i).deviceData, N, inputx_data.get(i).deviceData, K, beta, output_data.get(i).deviceData, N));
        // Copy the result from the device to the host
        context.copyDataToHost(output);
        context.clear();
    }

    //MK*KN
    public static void matmulBackward(Tensor inputx, Tensor inputy, Tensor output) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Tenserx inputx_data = context.getDeviceData(inputx);
        Tenserx inputy_data = context.getDeviceData(inputy);
        // Allocate Copy the memory from the host to the device
        Tenserx inputx_grad = context.getDeviceGrad(inputx);
        Tenserx inputy_grad = context.getDeviceGrad(inputy);
        Tenserx output_grad = context.getDeviceGrad(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(1), K = inputx.shape(2), N = inputy.shape(2);
        // KM = [KN * NM]
        forEach(inputx.shape(0), i -> cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, alpha, inputy_data.get(i).deviceData, N, output_grad.get(i).deviceData, N, beta, inputx_grad.get(i).deviceData, K));
        // Copy the result from the device to the host
        context.copyGradToHost(inputx);

        // NK = [NM * MK]
        forEach(inputx.shape(0), i -> cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, alpha, output_grad.get(i).deviceData, N, inputx_data.get(i).deviceData, K, beta, inputy_grad.deviceData, N));
        // Copy the result from the device to the host
        context.copyGradToHost(inputy);
        context.clear();
    }

    //MK*NK
    public static void matmulTranbForward(Tensor inputx, Tensor inputy, Tensor output, Tensor... alphas) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Tenserx inputx_data = context.getDeviceData(inputx);
        Tenserx inputy_data = context.getDeviceData(inputy);
        Tenserx output_data = context.getDeviceData(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(1), K = inputx.shape(2), N = inputy.shape(1);
        // NM = [NK * KM]
        forEach(inputx.shape(0), i -> cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, inputy_data.get(i).deviceData, K, inputx_data.get(i).deviceData, K, beta, output_data.get(i).deviceData, N));
        // Copy the result from the device to the host
        context.copyDataToHost(output);
        context.clear();
    }

    //MK*NK
    public static void matmulTranbBackward(Tensor inputx, Tensor inputy, Tensor output, Tensor... alphas) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Tenserx inputx_data = context.getDeviceData(inputx);
        Tenserx inputy_data = context.getDeviceData(inputy);
        // Allocate Copy the memory from the host to the device
        Tenserx inputx_grad = context.getDeviceGrad(inputx);
        Tenserx inputy_grad = context.getDeviceGrad(inputy);
        Tenserx output_grad = context.getDeviceGrad(output);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(1), K = inputx.shape(2), N = inputy.shape(1);
        // KM = [KN * NM]
        forEach(inputx.shape(0), i -> cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, alpha, inputy_data.get(i).deviceData, K, output_grad.get(i).deviceData, N, beta, inputx_grad.get(i).deviceData, K));
        // Copy the result from the device to the host
        context.copyGradToHost(inputx);

        // NK = [NM * MK]
        forEach(inputx.shape(0), i -> cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, alpha, inputx_data.get(i).deviceData, K, output_grad.get(i).deviceData, N, beta, inputy_grad.deviceData, K));
        // Copy the result from the device to the host
        context.copyGradToHost(inputy);
        context.clear();
    }

}