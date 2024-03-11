package com.deep.framework.cudnn;

import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.cudaStream_t;

import static com.deep.framework.cuda.Cuda.createDevicePointer;
import static com.deep.framework.cudnn.CudnnConfig.handle;
import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnActivationMode.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

public class Activation {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;
    private static final int DATA_TYPE_SZIE = Sizeof.DOUBLE;

    public static void sigmoidForward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SIGMOID);
        cudaStreamDestroy(stream);
    }

    public static void sigmoidBackward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SIGMOID);
        cudaStreamDestroy(stream);
    }

    public static void reluForward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_RELU);
        cudaStreamDestroy(stream);
    }

    public static void reluBackward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_RELU);
        cudaStreamDestroy(stream);
    }

    public static void tanhForward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_TANH);
        cudaStreamDestroy(stream);
    }

    public static void tanhBackward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_TANH);
        cudaStreamDestroy(stream);
    }

    public static void clippedReluForward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_CLIPPED_RELU);
        cudaStreamDestroy(stream);
    }

    public static void clippedReluBackward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_CLIPPED_RELU);
        cudaStreamDestroy(stream);
    }

    public static void eluForward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_ELU);
        cudaStreamDestroy(stream);
    }

    public static void eluBackward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_ELU);
        cudaStreamDestroy(stream);
    }

    public static void identityForward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_IDENTITY);
        cudaStreamDestroy(stream);
    }

    public static void identityBackward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_IDENTITY);
        cudaStreamDestroy(stream);
    }

    public static void swishForward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationForward(input.getData(), output.getData(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SWISH);
        cudaStreamDestroy(stream);
    }

    public static void swishBackward(Tensor input, Tensor output) {
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cudnnSetStream(handle, stream);
        activationBackward(input.getData(), input.getGrad(), output.getData(), output.getGrad(), Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SWISH);
        cudaStreamDestroy(stream);
    }

    public static void activationForward(double[] input, double[] output, int[] shape, int activation) {
        // 指定输入的维度
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        // 创建描述符句柄,指定输入描述符
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 创建输出数据和描述符句柄
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 创建激活函数描述符句柄
        cudnnActivationDescriptor activation_desc = new cudnnActivationDescriptor();
        cudnnCreateActivationDescriptor(activation_desc);
        cudnnSetActivationDescriptor(activation_desc, activation, CUDNN_NOT_PROPAGATE_NAN, 1);

        Pointer device_input = createDevicePointer(input);
        Pointer device_output = createDevicePointer(output);

        // 执行激活函数
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnActivationForward(handle, activation_desc, alpha, input_desc, device_input, beta, output_desc, device_output);
        cudaMemcpy(Pointer.to(output), device_output, output.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // 释放资源
        cudaFree(device_input);
        cudaFree(device_output);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyActivationDescriptor(activation_desc);
    }

    public static void activationBackward(double[] input, double[] input_grad, double[] output, double[] output_grad, int[] shape, int activation) {
        // 指定输入的维度
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        // 创建描述符句柄,指定输入描述符
        cudnnTensorDescriptor input_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(input_desc);
        cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 创建输出数据和描述符句柄
        cudnnTensorDescriptor output_desc = new cudnnTensorDescriptor();
        cudnnCreateTensorDescriptor(output_desc);
        cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, DATA_TYPE, batch_size, channels, height, width);

        // 创建激活函数描述符句柄
        cudnnActivationDescriptor activation_desc = new cudnnActivationDescriptor();
        cudnnCreateActivationDescriptor(activation_desc);
        cudnnSetActivationDescriptor(activation_desc, activation, CUDNN_NOT_PROPAGATE_NAN, 1);

        Pointer device_input = createDevicePointer(input);
        Pointer device_input_grad = createDevicePointer(input_grad);

        Pointer device_output = createDevicePointer(output);
        Pointer device_output_grad = createDevicePointer(output_grad);

        // 执行激活函数
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnActivationBackward(handle, activation_desc, alpha, output_desc, device_output, output_desc, device_output_grad, input_desc, device_input, beta, input_desc, device_input_grad);
        cudaMemcpy(Pointer.to(input_grad), device_input_grad, input_grad.length * DATA_TYPE_SZIE, cudaMemcpyDeviceToHost);

        // 释放资源
        cudaFree(device_input);
        cudaFree(device_input_grad);
        cudaFree(device_output);
        cudaFree(device_output_grad);

        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyActivationDescriptor(activation_desc);
    }
}