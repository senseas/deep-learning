package com.deep.framework.cudnn;

import com.deep.framework.cuda.CudaContext;
import com.deep.framework.graph.Tensor;
import com.deep.framework.lang.Shape;
import jcuda.Pointer;
import jcuda.jcudnn.cudnnActivationDescriptor;
import jcuda.jcudnn.cudnnHandle;
import jcuda.jcudnn.cudnnTensorDescriptor;

import static jcuda.jcudnn.JCudnn.*;
import static jcuda.jcudnn.cudnnActivationMode.*;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
import static jcuda.jcudnn.cudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

public class Activation {

    private static final int DATA_TYPE = CUDNN_DATA_DOUBLE;

    public static void sigmoidForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationForward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SIGMOID, context);
        context.clear();
    }

    public static void sigmoidBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationBackward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SIGMOID, context);
        context.clear();
    }

    public static void reluForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationForward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_RELU, context);
        context.clear();
    }

    public static void reluBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationBackward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_RELU, context);
        context.clear();
    }

    public static void tanhForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationForward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_TANH, context);
        context.clear();
    }

    public static void tanhBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationBackward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_TANH, context);
        context.clear();
    }

    public static void clippedReluForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationForward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_CLIPPED_RELU, context);
        context.clear();
    }

    public static void clippedReluBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationBackward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_CLIPPED_RELU, context);
        context.clear();
    }

    public static void eluForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationForward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_ELU, context);
        context.clear();
    }

    public static void eluBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationBackward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_ELU, context);
        context.clear();
    }

    public static void identityForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationForward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_IDENTITY, context);
        context.clear();
    }

    public static void identityBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationBackward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_IDENTITY, context);
        context.clear();
    }

    public static void swishForward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationForward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SWISH, context);
        context.clear();
    }

    public static void swishBackward(Tensor input, Tensor output) {
        CudaContext context = new CudaContext(output);
        activationBackward(input, output, Shape.shapes(input.getShape()), CUDNN_ACTIVATION_SWISH, context);
        context.clear();
    }

    public static void activationForward(Tensor input, Tensor output, int[] shape, int activation, CudaContext context) {
        // 指定输入的维度
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        cudnnHandle handle = context.getCudnnHandle();
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

        Pointer input_data = context.getDeviceData(input);
        Pointer output_data = context.getDeviceData(output);

        // 执行激活函数
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnActivationForward(handle, activation_desc, alpha, input_desc, input_data, beta, output_desc, output_data);
        context.copyDataToHost(output);

        // 释放资源
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyActivationDescriptor(activation_desc);
    }

    public static void activationBackward(Tensor input, Tensor output, int[] shape, int activation, CudaContext context) {
        // 指定输入的维度
        int batch_size = shape[0], channels = shape[1], height = shape[2], width = shape[3];

        cudnnHandle handle = context.getCudnnHandle();
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

        Pointer input_data = context.getDeviceData(input);
        Pointer input_grad = context.getDeviceGrad(input);

        Pointer output_data = context.getDeviceData(output);
        Pointer output_grad = context.getDeviceGrad(output);

        // 执行激活函数
        Pointer alpha = Pointer.to(new double[]{1}), beta = Pointer.to(new double[]{0});
        cudnnActivationBackward(handle, activation_desc, alpha, output_desc, output_data, output_desc, output_grad, input_desc, input_data, beta, input_desc, input_grad);
        context.copyGradToHost(input);

        // 释放资源
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyActivationDescriptor(activation_desc);
    }

}