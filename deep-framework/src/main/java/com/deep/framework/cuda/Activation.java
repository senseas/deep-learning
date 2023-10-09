package com.deep.framework.cuda;

import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cuda.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnActivationMode.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class Activation {

    public static void activationForward(double[] input_data, double[] output_data, int[] shape, int activation) {
        // 指定输入的维度
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        // 创建描述符句柄,指定输入描述符
        cudnnTensorDescriptor inputDesc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(inputDesc);
        cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, height, width);

        // 创建输出数据和描述符句柄
        cudnnTensorDescriptor outputDesc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(outputDesc);
        cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, height, width);

        // 创建激活函数描述符句柄
        cudnnActivationDescriptor activationDesc = new cudnnActivationDescriptor();
        cudnnCreateActivationDescriptor(activationDesc);
        cudnnSetActivationDescriptor(activationDesc, activation, CUDNN_NOT_PROPAGATE_NAN, 0);

        Pointer d_input_data = createDevicePointer(input_data);
        Pointer d_output_data = createDevicePointer(output_data);

        // 执行激活函数
        Pointer alpha = Pointer.to(new double[]{1.0f}), beta = Pointer.to(new double[]{0.0f});
        cudnnActivationForward(handle, activationDesc, alpha, inputDesc, d_input_data, beta, outputDesc, d_output_data);
        cudaMemcpy(Pointer.to(output_data), d_output_data, Shape.size(shape) * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        // 释放资源
        cudnnDestroyActivationDescriptor(activationDesc);
        cudnnDestroyTensorDescriptor(inputDesc);
        cudnnDestroyTensorDescriptor(outputDesc);
    }

    public static void activationBackward(double[] input_data, double[] input_grad_data, double[] output_data, double[] output_grad_data, int[] shape, int activation) {
        // 指定输入的维度
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        // 创建描述符句柄,指定输入描述符
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, height, width);

        // 创建描述符句柄,指定输入描述符
        cudnnTensorDescriptor input_grad_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_grad_desc);
        cudnnSetTensor4dDescriptor(input_grad_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, height, width);

        // 创建输出数据和描述符句柄
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, height, width);

        // 创建输出梯度数据和描述符句柄
        cudnnTensorDescriptor output_grad_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_grad_desc);
        cudnnSetTensor4dDescriptor(output_grad_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, batch_size, channels, height, width);

        // 创建激活函数描述符句柄
        cudnnActivationDescriptor activation_desc = new cudnnActivationDescriptor();
        cudnnCreateActivationDescriptor(activation_desc);
        cudnnSetActivationDescriptor(activation_desc, activation, CUDNN_NOT_PROPAGATE_NAN, 0);

        Pointer d_input_data = createDevicePointer(input_data);
        Pointer d_input_grad_data = createDevicePointer(input_grad_data);

        Pointer d_output_data = createDevicePointer(output_data);
        Pointer d_output_grad_data = createDevicePointer(output_grad_data);

        // 执行激活函数
        Pointer alpha = Pointer.to(new double[]{1.0f}), beta = Pointer.to(new double[]{0.0f});
        cudnnActivationBackward(handle, activation_desc, alpha, output_desc, d_output_data, output_grad_desc, d_output_grad_data, input_desc, d_input_data, beta, input_grad_desc, d_input_grad_data);
        cudaMemcpy(Pointer.to(input_grad_data), d_input_grad_data, Shape.size(shape) * Sizeof.DOUBLE, cudaMemcpyDeviceToHost);
        // 释放资源
        cudnnDestroyActivationDescriptor(activation_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
    }
}