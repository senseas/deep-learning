package com.deep.framework.cudnn;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnOpTensorDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class CudnnOpTensor {

    // 声明softmax算法和算法描述符
    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;

    public static void opTensor(double[] inputx, double[] inputy, double[] output, int[] shape, int op) {
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // Define input tensor
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define output tensor
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // Define op tensor
        cudnnOpTensorDescriptor op_tensor_desc = new cudnnOpTensorDescriptor();
        cudnnCreateOpTensorDescriptor(op_tensor_desc);
        cudnnSetOpTensorDescriptor(op_tensor_desc, op, DATA_TYPE, CUDNN_NOT_PROPAGATE_NAN);

        // allocate memory on device
        Pointer device_inputx = createDevicePointer(inputx);
        Pointer device_inputy = createDevicePointer(inputy);
        Pointer device_output = createDevicePointer(output);

        // Perform op operation
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnOpTensor(handle, op_tensor_desc, alpha, input_desc, device_inputx, alpha, input_desc, device_inputy, beta, output_desc, device_output);

        // copy device memory to host
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // clear up
        cudaFree(device_inputx);
        cudaFree(device_inputy);
        cudaFree(device_output);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyOpTensorDescriptor(op_tensor_desc);
    }

}