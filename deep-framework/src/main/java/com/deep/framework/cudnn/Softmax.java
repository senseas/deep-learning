package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.cudaStream_t;

import static com.deep.framework.cudnn.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_ACCURATE;
import static jcuda.jcudnn.cudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_INSTANCE;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaStreamCreate;
import static jcuda.runtime.JCuda.cudaStreamDestroy;

public class Softmax {

    // 声明softmax算法和算法描述符
    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int softmaxAlgo = CUDNN_SOFTMAX_ACCURATE;
    private static final int softmaxMode = CUDNN_SOFTMAX_MODE_INSTANCE;

    public static void softmaxForward(Tensor input, Tensor output, int[] shape) {
        shape = Shape.shapes(shape);
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);

        // 设置输入张量描述符
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        // 声明输入张量描述符
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 声明输出张量描述符
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 分配设备内存
        Pointer device_input = input.getDeviceData();
        Pointer device_output = output.getDeviceData();

        // 执行SoftmaxForward操作
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnSoftmaxForward(handle, softmaxAlgo, softmaxMode, alpha, input_desc, device_input, beta, output_desc, device_output);

        // 将输出数据复制到主机内存
        output.dataSynchronize();

        // 释放内存
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudaStreamDestroy(stream);
    }

    public static void softmaxBackward(Tensor input, Tensor output, int[] shape) {
        shape = Shape.shapes(shape);
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);

        // 设置输入张量描述符
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];
        // 声明输出张量描述符
        cudnnTensorDescriptor input_grad_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_grad_desc);
        cudnnSetTensor4dDescriptor(input_grad_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 声明输出张量描述符
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 分配设备内存
        Pointer device_output = output.getDeviceData();
        Pointer device_output_grad = output.getDeviceGrad();
        Pointer device_input_grad = input.getDeviceGrad();

        // 执行SoftmaxBackward操作
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnSoftmaxBackward(handle, softmaxAlgo, softmaxMode, alpha, output_desc, device_output, output_desc, device_output_grad, beta, input_grad_desc, device_input_grad);

        // 将输出数据复制到主机内存
        input.gradSynchronize();

        // 释放内存
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(input_grad_desc);
        cudaStreamDestroy(stream);
    }

}