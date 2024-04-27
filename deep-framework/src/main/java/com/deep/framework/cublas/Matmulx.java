package com.deep.framework.cublas;

import com.deep.framework.cuda.CudaContext;
import com.deep.framework.lang.Shape;
import com.deep.framework.lang.Tenserx;
import jcuda.Pointer;
import jcuda.jcublas.cublasHandle;

import static jcuda.jcublas.JCublas2.cublasDgemm;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;

public class Matmulx {

    //MK*KN
    public static void matmulForward(Tenserx inputx, Tenserx inputy, Tenserx output) {
        CudaContext context = new CudaContext(output);
        cublasHandle handle = context.getCublasHandle();

        // Allocate Copy the memory from the host to the device
        Pointer inputx_data = inputx.deviceData;
        Pointer inputy_data = inputy.deviceData;
        Pointer output_data = output.deviceData;

        // alpha, beta
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});

        int M = inputx.shape(0), K = inputx.shape(1), N = inputy.shape(1);
        // NM = [NK * KM]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, alpha, inputy_data, N, inputx_data, K, beta, output_data, N);
        // Copy the result from the device to the host
        context.copyDataToHost(output);
        context.clear();
    }

    public static void main(String[] args) {
        Tenserx A = new Tenserx(Shape.random(new int[]{2, 2, 3}), new int[]{2, 2, 3});
        Tenserx B = new Tenserx(Shape.random(new int[]{2, 3, 4}), new int[]{2, 3, 4});
        Tenserx C = new Tenserx(Shape.zeros(new int[]{2, 2, 4}), new int[]{2, 2, 4});
        matmulForward(A.get(0), B.get(0), C.get(0));
    }

}