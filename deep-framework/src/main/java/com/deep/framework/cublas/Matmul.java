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
        Pointer device_inputx = inputx.getDeviceData(deviceId, stream);
        Pointer device_inputy = inputy.getDeviceData(deviceId, stream);
        Pointer device_output = output.getDeviceData(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(1);
        // NM = [NK * KM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, device_inputy, N, device_inputx, K, beta, device_output, N);
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
        Pointer device_inputx = inputx.getDeviceData(deviceId, stream);
        Pointer device_inputy = inputy.getDeviceData(deviceId, stream);
        // Allocate Copy the memory from the host to the device
        Pointer device_inputx_grad = inputx.getDeviceGrad(deviceId, stream);
        Pointer device_inputy_grad = inputy.getDeviceGrad(deviceId, stream);
        Pointer device_output_grad = output.getDeviceGrad(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(1);
        // KM = KM_T[KN * NM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, alpha, device_inputy, N, device_output_grad, N, beta, device_inputx_grad, K);
        // Copy the result from the device to the host
        inputx.gradSynchronize(deviceId, stream);

        // NK = [NM * MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, alpha, device_output_grad, N, device_inputx, K, beta, device_inputy_grad, N);
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
        Pointer device_inputx = inputx.getDeviceData(deviceId, stream);
        Pointer device_inputy = inputy.getDeviceData(deviceId, stream);
        Pointer device_output = output.getDeviceData(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(0);
        // NM = [NK * KM]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, alpha, device_inputy, K, device_inputx, K, beta, device_output, N);
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
        Pointer device_inputx = inputx.getDeviceData(deviceId, stream);
        Pointer device_inputy = inputy.getDeviceData(deviceId, stream);
        // Allocate Copy the memory from the host to the device
        Pointer device_inputx_grad = inputx.getDeviceGrad(deviceId, stream);
        Pointer device_inputy_grad = inputy.getDeviceGrad(deviceId, stream);
        Pointer output_device_grad = output.getDeviceGrad(deviceId, stream);

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{alphas.length == 1 ? alphas[0].data() : 1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(0);
        // KM = KM_T[KN * NM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, alpha, device_inputy, K, output_device_grad, N, beta, device_inputx_grad, K);
        // Copy the result from the device to the host
        inputx.gradSynchronize(deviceId, stream);

        // NK = [GC_T=NM * MK]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, alpha, device_inputx, K, output_device_grad, N, beta, device_inputy_grad, K);
        // Copy the result from the device to the host
        inputy.gradSynchronize(deviceId, stream);

        cudaStreamDestroy(stream);
    }

}