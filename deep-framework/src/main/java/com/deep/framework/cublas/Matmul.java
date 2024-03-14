package com.deep.framework.cublas;

import com.deep.framework.graph.Tensor;
import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.cudaStream_t;

import static com.deep.framework.cublas.CublasConfig.getCublasHandle;
import static com.deep.framework.cuda.Cuda.createCudaStream;
import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.JCublas2.cublasSetStream;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaStreamDestroy;

public class Matmul {

    //MK*KN
    public static void matmulForward(Tensor inputx, Tensor inputy, Tensor output) {
        cublasHandle handle = getCublasHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cublasSetStream(handle, stream);

        // Allocate Copy the memory from the host to the device
        int deviceId = output.getDeviceId();
        Pointer inputx_data = inputx.getDeviceData(deviceId, stream);
        Pointer inputy_data = inputy.getDeviceData(deviceId, stream);
        Pointer output_data = output.getDeviceData(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(1);
        // NM = [NK * KM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, inputy_data, N, inputx_data, K, beta, output_data, N);
        // Copy the result from the device to the host
        output.dataSynchronize(deviceId, stream);

        cudaStreamDestroy(stream);
    }

    //MK*KN
    public static void matmulBackward(Tensor inputx, Tensor inputy, Tensor output) {
        cublasHandle handle = getCublasHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cublasSetStream(handle, stream);

        // Allocate Copy the memory from the host to the device
        int deviceId = output.getDeviceId();
        Pointer inputx_data = inputx.getDeviceData(deviceId, stream);
        Pointer inputy_data = inputy.getDeviceData(deviceId, stream);
        // Allocate Copy the memory from the host to the device
        Pointer inputx_grad = inputx.getDeviceGrad(deviceId, stream);
        Pointer inputy_grad = inputy.getDeviceGrad(deviceId, stream);
        Pointer output_grad = output.getDeviceGrad(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(1);
        // KM = [KN * NM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, alpha, inputy_data, N, output_grad, N, beta, inputx_grad, K);
        // Copy the result from the device to the host
        inputx.gradSynchronize(deviceId, stream);

        // NK = [NM * MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, alpha, output_grad, N, inputx_data, K, beta, inputy_grad, N);
        // Copy the result from the device to the host
        inputy.gradSynchronize(deviceId, stream);

        cudaStreamDestroy(stream);
    }

    //MK*NK
    public static void matmulTranbForward(Tensor inputx, Tensor inputy, Tensor output, Tensor... alphas) {
        cublasHandle handle = getCublasHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cublasSetStream(handle, stream);

        // Allocate Copy the memory from the host to the device
        int deviceId = output.getDeviceId();
        Pointer inputx_data = inputx.getDeviceData(deviceId, stream);
        Pointer inputy_data = inputy.getDeviceData(deviceId, stream);
        Pointer output_data = output.getDeviceData(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(0);
        // NM = [NK * KM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, inputy_data, K, inputx_data, K, beta, output_data, N);
        // Copy the result from the device to the host
        output.dataSynchronize(deviceId, stream);

        cudaStreamDestroy(stream);
    }

    //MK*NK
    public static void matmulTranbBackward(Tensor inputx, Tensor inputy, Tensor output, Tensor... alphas) {
        cublasHandle handle = getCublasHandle(output);
        cudaStream_t stream = createCudaStream(output);
        cublasSetStream(handle, stream);

        // Allocate Copy the memory from the host to the device
        int deviceId = output.getDeviceId();
        Pointer inputx_data = inputx.getDeviceData(deviceId, stream);
        Pointer inputy_data = inputy.getDeviceData(deviceId, stream);
        // Allocate Copy the memory from the host to the device
        Pointer inputx_grad = inputx.getDeviceGrad(deviceId, stream);
        Pointer inputy_grad = inputy.getDeviceGrad(deviceId, stream);
        Pointer output_grad = output.getDeviceGrad(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(0);
        // KM = [KN * NM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, alpha, inputy_data, K, output_grad, N, beta, inputx_grad, K);
        // Copy the result from the device to the host
        inputx.gradSynchronize(deviceId, stream);

        // NK = [NM * MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, alpha, inputx_data, K, output_grad, N, beta, inputy_grad, K);
        // Copy the result from the device to the host
        inputy.gradSynchronize(deviceId, stream);

        cudaStreamDestroy(stream);
    }

}